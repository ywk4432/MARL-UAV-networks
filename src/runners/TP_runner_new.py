from functools import partial
from multiprocessing import Pipe, Process

# from utils.plot_func import reward_plot
import numpy as np
import os
import random
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class TPRunner_new:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env_info = self.env.get_env_info(new=True)
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.step = 0
        self.t_env = 0
        self.step_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(
        self,
        l_scheme,
        l_group,
        l_preprocess,
        l_mac,
    ):
        self.new_l_batch = partial(
            EpisodeBatch,
            l_scheme,
            l_group,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=l_preprocess,
            device=self.args.device,
        )
        self.l_mac = l_mac
        self.l_scheme = l_scheme
        self.l_group = l_group
        self.l_preprocess = l_preprocess

    def get_env_info(self, info_type=None):
        return self.env.get_env_info(info_type)

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):

        self.env.reset()

        self.l_batch = self.new_l_batch()

        self.t = 0
        self.step = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = 0
        episode_lengths = 0
        # self.l_mac.init_hidden(batch_size=self.batch_size)
        # self.f_mac.init_hidden(batch_size=self.batch_size)
        terminated = False
        # envs_not_terminated = [
        #     b_idx for b_idx, termed in enumerate(terminated) if not termed
        # ]
        final_env_infos = []

        slot_step = 0
        while True:
            if slot_step == 0:
                # 聚类
                self.env.bkm()

                if self.args.env_args["pointer_match"]:
                    # obs-bkm结果
                    l_pre_transition_data = {
                        "state": [],
                        "avail_actions": [],
                        "obs": [],
                    }

                    data = {
                        "l_state": self.env.get_state(),
                        "l_avail_actions": self.env.get_l_avail_actions(),
                        "l_obs": self.env.get_luav_obs_new(),
                    }

                    l_pre_transition_data["state"].append(data["l_state"])
                    l_pre_transition_data["avail_actions"].append(
                        data["l_avail_actions"]
                    )
                    l_pre_transition_data["obs"].append(data["l_obs"])

                    self.l_batch.update(l_pre_transition_data, ts=self.t)

                    # 确定领导者无人机的动作
                    l_actions = self.l_mac.select_actions(
                        self.l_batch,
                        t_ep=self.t,
                        t_env=self.t_env,
                        # bs=envs_not_terminated,
                        test_mode=test_mode,
                    )
                    l_cpu_actions = l_actions.to("cpu").numpy()
                    l_actions_chosen = {
                        "actions": l_actions.unsqueeze(1).to("cpu"),
                    }
                    self.l_batch.update(
                        l_actions_chosen,
                        # bs=envs_not_terminated,
                        ts=self.t,
                        mark_filled=False,
                    )

                    (
                        l_reward,
                        f_reward_agents,
                        f_reward,
                        slot_end,
                        terminated,
                        env_info,
                    ) = self.env.step(slot_step, l_cpu_actions)

                    # Return the observations, avail_actions and state to make the next action
                    data = {
                        # Rest of the data for the current timestep
                        "l_reward": l_reward,
                        "l_agent_reward": l_reward,
                        "terminated": terminated,
                        "info": env_info,
                        "slot_end": slot_end,
                    }

                    l_post_transition_data = {
                        "reward": [],
                        "terminated": [],
                        "agent_reward": [],
                    }

                    l_post_transition_data["reward"].append((data["l_reward"],))
                    l_post_transition_data["agent_reward"].append(
                        (data["l_agent_reward"],)
                    )

                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True

                    l_post_transition_data["terminated"].append((env_terminated,))

                    self.l_batch.update(
                        l_post_transition_data,
                        # bs=envs_not_terminated,
                        ts=self.t,
                        mark_filled=False,
                    )

                    # self.step += 1
                    # self.l_batch.update(
                    #     l_pre_transition_data,
                    #     # bs=envs_not_terminated,
                    #     ts=self.t,
                    #     mark_filled=True,
                    # )
                else:
                    print("ERROR!")
                    exit(0)

                slot_step += 1

            elif slot_step >= 1 and slot_step < self.args.env_args["slot_step_num"]:
                # TODO:

                self.env.ra(slot_step)

                slot_step += 1

            elif slot_step == self.args.env_args["slot_step_num"]:

                self.env.ra(slot_step)

                env_terminated = False
                if data["terminated"]:
                    final_env_infos.append(data["info"])
                if data["terminated"] and not data["info"].get("episode_limit", False):
                    env_terminated = True

                self.t += 1
                self.step += 1

                slot_step = 0

            else:
                assert slot_step <= self.args.env_args["slot_step_num"]

            if terminated:
                break

        if test_mode:
            # if os.getpid() == self.ps[0].pid:
            # self.env.record(t_env=self.t_env)
            path = f"record/{self.args.unique_token}"
            self.env.record(path=path, t_env=self.t_env)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        env_stats = self.env.get_env_info()

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        # self.batch_size = 1
        # episode_lengths 也是0？

        # cur_returns.extend(episode_returns)
        # 一直是0？
        # episode_return += reward

        # n_test_runs = (
        #     max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        # )
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.l_mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.l_mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.l_batch

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
