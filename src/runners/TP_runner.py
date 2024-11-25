from functools import partial
from multiprocessing import Pipe, Process

# from utils.plot_func import reward_plot
import numpy as np
import os
import random
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class TPRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env_info = self.env.get_env_info()
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
        f_scheme,
        l_group,
        f_group,
        l_preprocess,
        f_preprocess,
        l_mac,
        f_mac,
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
        self.new_f_batch = partial(
            EpisodeBatch,
            f_scheme,
            f_group,
            self.batch_size,
            self.args.env_args["slot_step_num"] * self.episode_limit + 1,
            preprocess=f_preprocess,
            device=self.args.device,
        )
        self.l_mac = l_mac
        self.f_mac = f_mac
        self.l_scheme = l_scheme
        self.f_scheme = f_scheme
        self.l_group = l_group
        self.f_group = f_group
        self.l_preprocess = l_preprocess
        self.f_preprocess = f_preprocess

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):

        self.env.reset()

        self.l_batch = self.new_l_batch()
        self.f_batch = self.new_f_batch()

        self.t = 0
        self.step = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = 0
        episode_lengths = 0
        # self.l_mac.init_hidden(batch_size=self.batch_size)
        self.f_mac.init_hidden(batch_size=self.batch_size)
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
                        "l_obs": self.env.get_luav_obs(),
                    }

                    l_pre_transition_data["state"].append(data["l_state"])
                    l_pre_transition_data["avail_actions"].append(
                        data["l_avail_actions"]
                    )
                    l_pre_transition_data["obs"].append(data["l_obs"])

                    self.l_batch.update(l_pre_transition_data, ts=0)

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
                    self.l_batch.update(
                        l_pre_transition_data,
                        # bs=envs_not_terminated,
                        ts=self.t,
                        mark_filled=True,
                    )
                else:
                    (
                        l_reward,
                        f_reward_agents,
                        f_reward,
                        slot_end,
                        terminated,
                        env_info,
                    ) = self.env.step(slot_step, None)

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

        return self.l_batch, None

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


def env_worker(remote, env_fn):
    """用于在子进程中运行的环境工作函数，它通过与主进程之间的管道通信，执行了一些基本的环境操作。该函数接收两个参数：remote 和 env_fn。

    Args:
        remote (_type_): multiprocessing.Pipe 对象，用于在子进程和主进程之间进行双向通信。通过这个管道，子进程可以接收主进程发送的命令和数据，并且可以将处理结果发送回主进程。
        env_fn (_type_): 函数，用于创建环境的实例。

    Raises:
        NotImplementedError: _description_
    """
    # Make environment
    # 调用 env_fn.x()，子进程在开始时会创建一个新的环境实例。
    env = env_fn.x()
    # 函数主体是一个无限循环，不断等待主进程发送的命令并执行相应的操作。
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            # 如果 cmd == "step"，则表示主进程要求子进程执行一个时间步的环境步骤。
            slot_step = data[0]
            uav_type = data[1]
            uav_actions = data[2]
            # Take a step in the environment
            # 子进程通过调用环境的 step 方法执行动作，并返回当前状态、可用动作、观测、奖励、终止标志和环境信息。
            l_reward, f_reward_agents, f_reward, slot_end, terminated, env_info = (
                env.step_dmtd(slot_step, uav_type, uav_actions)
            )
            # Return the observations, avail_actions and state to make the next action
            l_state = env.get_state()
            f_state = env.get_state()
            l_avail_actions = env.get_l_avail_actions()
            f_avail_actions = env.get_f_avail_actions()
            l_obs = env.get_luav_obs()
            f_obs = env.get_fuav_obs()
            # 这些数据被封装成一个字典，并通过管道发送给主进程。
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "l_state": l_state,
                    "l_avail_actions": l_avail_actions,
                    "l_obs": l_obs,
                    # Rest of the data for the current timestep
                    "l_reward": l_reward,
                    "l_agent_reward": l_reward,
                    "f_state": f_state,
                    "f_avail_actions": f_avail_actions,
                    "f_obs": f_obs,
                    # Rest of the data for the current timestep
                    "f_reward": f_reward,
                    "f_agent_reward": f_reward_agents,
                    "terminated": terminated,
                    "info": env_info,
                    "slot_end": slot_end,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "l_state": env.get_state(),
                    "l_avail_actions": env.get_l_avail_actions(),
                    "l_obs": env.get_luav_obs(),
                    "f_state": env.get_state(),
                    "f_avail_actions": env.get_f_avail_actions(),
                    "f_obs": env.get_fuav_obs(),
                }
            )
        elif cmd == "close":
            # env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_state())
        elif cmd == "record":
            if os.getpid() == data[0]:
                env.record(t_env=data[1])
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
