import copy

import torch as th
from torch.optim.rmsprop import RMSprop
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from modules.critics.pointer_critic import StateCritic
from utils.rl_utils import (
    build_td_lambda_targets,
    build_target_q,
    build_n_step_td_targets,
    build_q_lambda_targets,
)
from utils.th_utils import get_parameters_num


class Learner:
    def __init__(self, mac, scheme, logger, args, env_info):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.env_info = env_info

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = StateCritic(
            self.env_info["l_s_obs_shape"],
            self.env_info["l_d_obs_shape"],
            self.args.l_args["hidden_size"],
            self.args.env_args["fuav_num"],
        )  ##
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params
        self.c_params = self.critic_params
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.static_size = env_info["l_s_obs_shape"]
        self.dynamic_size = env_info["l_d_obs_shape"]
        self.sequence_size = self.args.env_args["fuav_num"]
        self.batch_size = self.args.l_args["batch_size"]
        self.episode_limit = self.args.env_args["episode_limit"]

        # print('Mixer Size: ')
        print(get_parameters_num(list(self.c_params)))

    def prepare_inputs(self, inputs):
        # 转换inputs

        static = inputs[:, :, :, : self.static_size].reshape(
            self.batch_size,
            int(self.static_size / self.sequence_size),
            self.sequence_size,
        )
        dynamic = inputs[:, :, :, self.static_size :].reshape(
            self.batch_size,
            int(self.dynamic_size / self.sequence_size),
            self.sequence_size,
        )
        return static, dynamic

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :]

        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        advantages, td_error, targets_taken, log_pi_taken = self._calculate_advs(
            batch, rewards, terminated, actions, avail_actions, critic_mask, bs, max_t
        )

        # pg_loss = -((advantages.detach() * log_pi_taken) * mask).sum() / mask.sum()
        pg_loss = ((advantages.detach() * log_pi_taken) * mask).sum() / mask.sum()
        vf_loss = ((td_error**2) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_optimiser.zero_grad()
        vf_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(
                "critic_loss",
                ((td_error**2) * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "td_error_abs",
                (td_error.abs() * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "q_taken_mean",
                (targets_taken * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                ((targets_taken + advantages) * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "pg_loss",
                # -((advantages.detach() * log_pi_taken) * mask).sum().item()
                # / mask.sum().item(),
                ((advantages.detach() * log_pi_taken) * mask).sum().item()
                / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def _calculate_advs(
        self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t
    ):
        mac_out = []
        q_outs = []
        action_outs = []

        for t in range(batch.max_seq_length):
            agent_out = self.mac.forward(batch, t=t)
            inputs = self.critic._build_inputs(batch, bs, t)
            static, dynamic = self.prepare_inputs(inputs)
            q_out = self.critic.forward(
                static,
                dynamic,
            )
            action_outs.append(agent_out[0])
            mac_out.append(agent_out[1])
            q_outs.append(q_out)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        q_outs = th.stack(q_outs, dim=1)  # Concat over time

        # Calculated baseline
        pi = mac_out[:, :-1]  # [bs, t, n_agents, n_actions]
        pi = pi.unsqueeze(2)
        pi_taken = th.gather(pi, dim=-1, index=actions[:, :-1]).squeeze(
            -1
        )  # [bs, t, n_agents]
        action_mask = mask.repeat(1, 1, self.n_agents)
        pi_taken[action_mask == 0] = 1.0
        pi_taken = pi_taken.sum(dim=3)
        log_pi_taken = pi_taken.reshape(-1)

        # Calculate q targets
        targets_taken = q_outs  # [bs, t, n_agents]

        if getattr(self.args, "n_td", False):
            targets = build_n_step_td_targets(
                rewards,
                terminated,
                mask,
                targets_taken,
                self.args.n_agents,
                self.args.gamma,
                self.args.n_step,
            )
        else:
            targets = build_td_lambda_targets(
                rewards,
                terminated,
                mask,
                targets_taken,
                self.args.n_agents,
                self.args.gamma,
                self.args.td_lambda,
            )

        advantages = targets - targets_taken[:, :-1]
        advantages = advantages.unsqueeze(2).repeat(1, 1, self.n_agents, 1).reshape(-1)

        td_error = targets_taken[:, :-1] - targets.detach()
        td_error = td_error.unsqueeze(2).repeat(1, 1, self.n_agents, 1).reshape(-1)

        return (
            advantages,
            td_error,
            targets_taken[:, :-1]
            .unsqueeze(2)
            .repeat(1, 1, self.n_agents, 1)
            .reshape(-1),
            log_pi_taken,
        )

    def build_exp_q(self, target_q_vals, mac_out, states):
        target_exp_q_vals = th.sum(target_q_vals * mac_out, dim=3)
        return target_exp_q_vals

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
