import copy
from pathlib import Path

import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from utils.plot_func import write_csv
from utils.rl_utils import (
    build_td_lambda_targets,
    build_q_lambda_targets,
    build_n_step_td_targets,
)
from utils.th_utils import get_parameters_num


class Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device("cuda" if args.use_cuda else "cpu")
        self.params = list(mac.parameters())

        print("Only Agent local Update: ")
        print("Agent Size: ")
        print(get_parameters_num(self.mac.agent.parameters()))

        if self.args.optimizer == "adam":
            self.optimiser = Adam(
                params=self.params,
                lr=args.lr,
                weight_decay=getattr(args, "weight_decay", 0),
            )
        else:
            self.optimiser = RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, "use_per", False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float("-inf")
            self.priority_min = float("inf")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        agent_rewards = batch["agent_reward"][
            :, :-1
        ]  # 选择 x 的第二个轴（axis=1）上的所有元素，但是不包括最后一个元素，因此是从第一个时间步到倒数第二个时间步的数据
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            if getattr(self.args, "q_lambda", False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)

                targets = build_q_lambda_targets(
                    agent_rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    qvals,
                    self.args.gamma,
                    self.args.td_lambda,
                )
            elif getattr(self.args, "n_td", False):
                targets = build_n_step_td_targets(
                    agent_rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.n_step,
                )
            else:
                targets = build_td_lambda_targets(
                    agent_rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.td_lambda,
                )

        td_error = chosen_action_qvals - targets.detach()
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_agent_targets()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td_agent", L_td.item(), t_env)
            self.logger.log_stat("grad_norm_agent", grad_norm, t_env)
            mask_elems = mask.sum().item()
            td_error_abs = masked_td_error.abs().sum().item() / mask_elems
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)

            # TODO: 修改保存路径
            """
            # 记录各训练批次智能体的TD误差
                if self.args.runner == "two_timescale":
                    path = Path(f"record/Loss/{self.args.name}/{self.args.os_pid}")
                else:
                    path = Path(f"record/Loss/{self.args.os_pid}")
                if not path.exists():
                    path.mkdir(parents=True)
                path = path / "td_error_abs_agent.csv"
                write_csv(path, [td_error_abs])
            """
            self.logger.log_stat(
                "q_taken_mean_agent",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean_agent",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = agent_rewards.sum(1).detach().to("cpu")
                # normalize to [0, 1]
                self.priority_max = max(
                    th.max(info["td_errors_abs"]).item(), self.priority_max
                )
                self.priority_min = min(
                    th.min(info["td_errors_abs"]).item(), self.priority_min
                )
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) / (
                    self.priority_max - self.priority_min + 1e-5
                )
            else:
                info["td_errors_abs"] = (
                    ((td_error.abs() * mask).sum(1) / th.sqrt(mask.sum(1)))
                    .detach()
                    .to("cpu")
                )
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def _update_agent_targets(self):
        self.target_mac.load_state(self.mac)
        # self.logger.console_logger.info("Updated target network, only agent")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
