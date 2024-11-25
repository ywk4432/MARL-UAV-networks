from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class MultiObjectRunner:

    def __init__(self, args, logger):
        self.batch = None
        self.mac = None
        self.new_batch = None
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.args.env_args["slot_num"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self, path):
        self.env.save_replay(path)

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def get_state(self):
        state = self.env.get_state()
        state = state[1] if self.args.agent_type == "high" else state[0]
        return np.array([state])

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "obs": [self.get_state()],
                "avail_actions": [np.ones(self.args.n_actions)],
            }
            pre_transition_data["state"] = pre_transition_data["obs"]

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            terminated = self.env.step(actions[0][0])
            rewards = self.env.get_reward()
            if self.args.agent_type == "expert":
                reward = rewards[self.args.agent_id]
            elif self.args.agent_type == "preference":
                reward = np.sum(rewards[:3] * self.args.preference)
            else:
                reward = rewards[-1]
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "agent_reward": [(reward,)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.get_state()],
            "avail_actions": [np.ones(self.args.n_actions)],
        }
        last_data["obs"] = last_data["state"]
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()
