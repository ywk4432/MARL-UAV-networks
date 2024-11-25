import collections
import datetime
import os
import pickle
import pprint
import threading
import time
from os.path import dirname, abspath
import pathlib
from types import SimpleNamespace as SN

import numpy as np
import torch as th

from components.episode_buffer import PrioritizedReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(
        args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def get_preference_set(m: int, d: int):
    allocation = []

    def dfs(res: list, slots: int, total: int, pos=0, allocated=0):
        if pos == slots:
            if allocated == total:
                allocation.append(res.copy())
            return
        for i in range(0, total - allocated + 1):
            res.append(i)
            dfs(res, slots, total, pos + 1, allocated + i)
            del res[-1]

    dfs([], m, d)
    return np.array(allocation) / d


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = 1
    preference = get_preference_set(args.expert_num, args.preference_d)[args.agent_id]
    args.preference = preference
    if args.agent_type == "high":
        args.n_actions = len(preference)
        args.state_shape = env_info["state_shape"][1]
        args.agent = args.high_agent
    else:
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"][0]
    args.obs_shape = args.state_shape
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, "agent_own_state_size", False):
        print("智能体无自己的观测尺寸")

    # Default/Base scheme
    scheme = {
        "obs": {"vshape": args.state_shape},
        "state": {"vshape": args.state_shape},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (args.n_actions,),
            "group": "agents",
            "dtype": th.int,
        },
        "probs": {
            "vshape": (args.n_actions,),
            "group": "agents",
            "dtype": th.float,
        },
        "agent_reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    if args.agent_type == "preference":
        args.buffer_size *= args.expert_num + 1
    buffer = PrioritizedReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        args.env_args["slot_num"] + 1,
        args.per_alpha,
        args.per_beta,
        args.t_max,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    if args.agent_type == "preference":
        # 经验迁移
        path = f"{args.local_results_path}/expert_experience"
        for i in range(args.expert_num):
            with open(f"{path}/{i}.pickle", "rb") as f:
                experience = pickle.load(f)
                buffer.insert_episode_batch(experience)
        # 模型迁移
        expert_state_dicts = [
            th.load(
                f"{args.local_results_path}/models/expert_{i}/agent.th",
                map_location=lambda storage, loc: storage,
            )
            for i in range(args.expert_num)
        ]
        state_dict = collections.OrderedDict()
        for key in mac.agent.state_dict():
            state_dict[key] = sum(
                preference[i] * expert_state_dicts[i][key] for i in args.expert_num
            )
        mac.agent.load_state_dict(state_dict)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            # 通过 buffer.sample(args.batch_size) 从缓存中抽样一个批次的数据
            episode_sample, idx, weights = buffer.sample(
                args.batch_size, runner.t_env
            )

            # 将批次数据进行处理，确保只使用填充数据的有效时间步，通过 episode_sample = episode_sample[:, :max_ep_t] 进行截断
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 将批次数据传递给 learner.train 进行训练
            # logger.console_logger.info("神经网络更新")
            info = learner.train(
                episode_sample, runner.t_env, episode, weights
            )
            del episode_sample
            info["td_errors_abs"] = info["td_errors_abs"].sum(dim=1)
            new_priorities = info["td_errors_abs"].flatten() + 1e-6
            buffer.update_priorities(idx, new_priorities.numpy().tolist())

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
                runner.save_replay(
                    f"{args.local_results_path}/replay/{args.agent_type}_{args.agent_id}/{runner.t_env}"
                )

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = (
                f"{args.local_results_path}/models/{args.agent_type}_{args.agent_id}"
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)
            # save_path/: agent.th, opt.th

            if args.agent_type == "expert":
                if buffer.can_sample(args.buffer_size):
                    logger.console_logger.info("Save Experience")
                    path = pathlib.Path(f"{args.local_results_path}/expert_experience")
                    if not path.exists():
                        path.mkdir(parents=True)
                    with open(path / f"{args.agent_id}.pickle", "wb") as f:
                        pickle.dump(buffer, f)
                else:
                    logger.console_logger.info("Demonstrations Not Full")

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
