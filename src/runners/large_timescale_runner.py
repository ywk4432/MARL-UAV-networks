import os
from functools import partial
from multiprocessing import Pipe, Process

# from utils.plot_func import reward_plot
import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class LargeTimsecaleRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **self.args.env_args)),
                ),
            )
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
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

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.l_batch = self.new_l_batch()

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        l_pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            l_pre_transition_data["state"].append(data["l_state"])
            l_pre_transition_data["avail_actions"].append(data["l_avail_actions"])
            l_pre_transition_data["obs"].append(data["l_obs"])

        self.l_batch.update(l_pre_transition_data, ts=0)

        self.t = 0
        self.step = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.l_mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = []

        while True:
            l_actions = self.l_mac.select_actions(
                self.l_batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )
            l_cpu_actions = l_actions.to("cpu").numpy()
            l_actions_chosen = {
                "actions": l_actions.unsqueeze(1).to("cpu"),
            }
            self.l_batch.update(
                l_actions_chosen,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(
                            (
                                "step",
                                (
                                    0,
                                    "l",
                                    l_cpu_actions[action_idx],
                                ),
                            )
                        )
                    action_idx += 1

            l_post_transition_data = {
                "reward": [],
                "terminated": [],
                "agent_reward": [],
            }
            l_pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
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
                    terminated[idx] = data["terminated"]

                    l_post_transition_data["terminated"].append((env_terminated,))
                    l_pre_transition_data["state"].append(data["l_state"])
                    l_pre_transition_data["avail_actions"].append(
                        data["l_avail_actions"]
                    )
                    l_pre_transition_data["obs"].append(data["l_obs"])

            self.l_batch.update(
                l_post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            self.t += 1

            self.l_batch.update(
                l_pre_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=True,
            )

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

        if test_mode:
            # print_reward_list = []
            # for idx in range(self.batch_size):
            #     print_reward_list.append(sum(reward_list[idx]) / len(reward_list[idx]))
            # print(
            #     f"=== t:{self.t_env} env_idx:{idx} reward:{np.mean(print_reward_list)} ==="
            # )
            # reward_list = [[] for _ in range(self.batch_size)]

            for parent_conn in self.parent_conns:
                parent_conn.send(("record", (self.ps[0].pid, self.t_env)))

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

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


def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            slot_step = data[0]
            uav_type = data[1]
            uav_actions = data[2]
            l_reward, f_reward_agents, f_reward, slot_end, terminated, env_info = (
                env.step(slot_step, uav_type, uav_actions)
            )
            l_state = env.get_state()
            l_avail_actions = env.get_l_avail_actions()
            l_obs = env.get_luav_obs()
            remote.send(
                {
                    "l_state": l_state,
                    "l_avail_actions": l_avail_actions,
                    "l_obs": l_obs,
                    "l_reward": l_reward,
                    "l_agent_reward": l_reward,
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
                }
            )
        elif cmd == "close":
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