import os
from functools import partial
from multiprocessing import Pipe, Process
from pathlib import Path

# from utils.plot_func import reward_plot
import numpy as np
import torch as th

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class TransferRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        # 根据 self.args.env 获取一个环境函数1000
        env_fn = env_REGISTRY[self.args.env]
        # self.ps 用于存储后续创建的子进程对象
        self.ps = []
        # 使用 enumerate 遍历 self.worker_conns 元组中的每个子进程的通信管道和索引
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **self.args.env_args)),
                ),
            )
            # 将创建的子进程对象添加到 self.ps 列表中
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        # 使用 multiprocessing 模块的管道（Pipe）来实现主进程与子进程之间的通信
        self.parent_conns[0].send(("get_env_info", None, None))

        self.env_info = self.parent_conns[0].recv()

        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

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
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None, None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None, None))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        final_env_infos = []

        save_probs = getattr(self.args, "save_probs", False)
        reward_print_env_id = 1
        reward_list = [[] for _ in range(self.batch_size)]
        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )
            else:
                actions = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    # Only send the actions to the env if it hasn't terminated
                    if not terminated[idx]:
                        parent_conn.send(
                            ("step", cpu_actions[action_idx], self.args.model_type)
                        )
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": [], "agent_reward": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["agent_reward"].append((data["agent_reward"],))

                    if test_mode:
                        reward_list[idx].append(data["agent_reward"])

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if test_mode:
            # fig_name = (
            #     f"fig/{self.args.mixer}-{self.args.learner}-{self.args.td_lambda}"
            # )
            # reward_plot(reward_list, fig_name, self.batch_size)
            print_reward_list = []
            for idx in range(self.batch_size):
                print_reward_list.append(sum(reward_list[idx]) / len(reward_list[idx]))

            print(
                f"=== t:{self.t_env} env_idx:{idx} reward:{np.mean(print_reward_list)} ==="
            )

            reward_list = [[] for _ in range(self.batch_size)]

            for parent_conn in self.parent_conns:
                parent_conn.send(
                    (
                        "record",
                        (self.ps[0].pid, f"{self.args.run_id}"),
                        None,
                    )
                )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None, None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

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
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        if self.args.model_save and self.t_env % self.args.save_interval == 0:
            model_save_path = Path(f"record/{self.args.run_id}/suboptimal_agents")
            if not model_save_path.exists():
                model_save_path.mkdir(parents=True)
            model_save_path /= f"{self.args.model_type}.pth"
            th.save(self.mac.agent.state_dict(), model_save_path)

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
        cmd, data, model_type = remote.recv()
        if cmd == "step":
            # 如果 cmd == "step"，则表示主进程要求子进程执行一个时间步的环境步骤。
            actions = data
            # Take a step in the environment
            # 子进程通过调用环境的 step 方法执行动作，并返回当前状态、可用动作、观测、奖励、终止标志和环境信息。
            rewards_agents, reward, terminated, env_info = env.step(
                actions, model_type=model_type
            )
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            # 这些数据被封装成一个字典，并通过管道发送给主进程。
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "agent_reward": rewards_agents,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            # env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "record":
            if os.getpid() == data[0]:
                env.record(path=f"record/{data[1]}")
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