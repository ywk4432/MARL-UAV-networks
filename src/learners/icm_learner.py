import copy
from pathlib import Path

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from utils.plot_func import write_csv
from utils.rl_utils import (
    build_td_lambda_targets,
    build_q_lambda_targets,
    build_n_step_td_targets,
)
from utils.th_utils import get_parameters_num


class ICM(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(ICM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

        self.pred_module1 = nn.Linear(512 + num_actions, 256)
        self.pred_module2 = nn.Linear(256, 512)

        self.invpred_module1 = nn.Linear(512 + 512, 256)
        self.invpred_module2 = nn.Linear(256, num_actions)

    def get_feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return x

    def forward(self, x):
        # get feature
        feature_x = self.get_feature(x)
        return feature_x

    def get_full(self, x, x_next, a_vec):
        # get feature
        feature_x = self.get_feature(x)
        feature_x_next = self.get_feature(x_next)

        pred_s_next = self.pred(feature_x, a_vec)  # predict next state feature
        pred_a_vec = self.invpred(feature_x, feature_x_next)  # (inverse) predict action

        return pred_s_next, pred_a_vec, feature_x_next

    def pred(self, feature_x, a_vec):
        # Forward prediction: predict next state feature, given current state feature and action (one-hot)
        pred_s_next = F.relu(
            self.pred_module1(th.cat([feature_x, a_vec.float()], dim=-1).detach())
        )
        pred_s_next = self.pred_module2(pred_s_next)
        return pred_s_next

    def invpred(self, feature_x, feature_x_next):
        # Inverse prediction: predict action (one-hot), given current and next state features
        pred_a_vec = F.relu(
            self.invpred_module1(th.cat([feature_x, feature_x_next], dim=-1))
        )
        pred_a_vec = self.invpred_module2(pred_a_vec)
        return F.softmax(pred_a_vec, dim=-1)


class Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device("cuda" if args.use_cuda else "cpu")
        self.params = list(mac.parameters())

        self.learn_count = 0

        if args.mixer == "qmix":
            self.mixer = Mixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "local":
            self.mixer = None
        else:
            raise "mixer error"

        if self.mixer is not None:
            self.target_mixer = copy.deepcopy(self.mixer)
            self.params += list(self.mixer.parameters())

            print(
                f"{args.name} Mixer Size: {get_parameters_num(self.mixer.parameters())}"
            )

        print(
            f"{args.name} Agent Size: {get_parameters_num(self.mac.agent.parameters())}"
        )

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
        self.ICM = ICM(in_channels=args.in_channels, num_actions=args.n_actions)

        # priority replay
        self.use_per = getattr(self.args, "use_per", False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float("-inf")
            self.priority_min = float("inf")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        self.learn_count += 1
        # Get the relevant quantities
        rewards = batch["reward"][
            :, :-1
        ]  # 选择 x 的第二个轴（axis=1）上的所有元素，但是不包括最后一个元素，因此是从第一个时间步到倒数第二个时间步的数据
        actions = batch["actions"][:, :-1]
        obs = batch["obs"][:, :-2]
        next_obs = batch["obs"][:, 1:-1]
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

        # get ICM results
        a_vec = F.one_hot(
            actions, num_classes=self.args.n_actions
        )  # convert action from int to one-hot format
        pred_s_next, pred_a_vec, feature_x_next = self.ICM.get_full(
            obs, next_obs, a_vec
        )
        # calculate forward prediction and inverse prediction loss
        forward_loss = F.mse_loss(
            pred_s_next, feature_x_next.detach(), reduction="none"
        )
        inverse_pred_loss = F.cross_entropy(
            pred_a_vec, actions.detach(), reduction="none"
        )

        # calculate rewards
        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1)
        total_rewards = intrinsic_rewards.clone()
        if self.use_extrinsic:
            total_rewards += rewards

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
            if self.mixer is not None:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, "q_lambda", False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(
                    total_rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    qvals,
                    self.args.gamma,
                    self.args.td_lambda,
                )
            elif getattr(self.args, "n_td", False):
                targets = build_n_step_td_targets(
                    total_rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.n_step,
                )
            else:
                targets = build_td_lambda_targets(
                    total_rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.td_lambda,
                )

        # Mixer
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
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
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            td_error_abs = masked_td_error.abs().sum().item() / mask_elems
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("learn_count", self.learn_count, t_env)

            # 记录各训练批次智能体的TD误差
            if self.args.runner == "two_timescale":
                path = Path(f"record/Loss/{self.args.name}/{self.args.os_pid}")
            else:
                path = Path(f"record/Loss/{self.args.os_pid}")
            if not path.exists():
                path.mkdir(parents=True)
            path = path / "td_error_abs_agent.csv"
            write_csv(path, [td_error_abs])

            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to("cpu")
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
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
