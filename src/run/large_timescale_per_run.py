import copy
import datetime
import os
import pprint
import threading
import time
from os.path import dirname, abspath
from types import SimpleNamespace as SN

import torch as th

from components.episode_buffer import PrioritizedReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)
    _config["use_per"] = True
    _config["return_priority"] = True

    _l_config = copy.deepcopy(_config)

    for k in _config["l_args"]:
        _l_config[k] = _config["l_args"][k]

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    l_args = SN(**_l_config)
    l_args.device = "cuda" if l_args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info(f"主进程ID：{os.getpid()}, Default Sample")
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(
        args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    args.unique_token = unique_token
    if args.use_tensorboard:
        _log.info("=== * 使用 tensorboard * ===")
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger, l_args=l_args)

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


def run_sequential(args, logger, l_args):
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.l_n_agents = env_info["l_n_agents"]
    args.l_n_actions = env_info["l_n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    args.os_pid = os.getpid()

    # 设置领航无人机参数信息
    l_args.n_agents = env_info["l_n_agents"]
    l_args.n_actions = env_info["l_n_actions"]
    l_args.state_shape = env_info["state_shape"]
    l_args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    l_args.os_pid = os.getpid()

    if getattr(args, "agent_own_state_size", False):
        print("智能体无自己的观测尺寸")

    # Default/Base scheme
    l_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["l_obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["l_n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "probs": {
            "vshape": (env_info["l_n_actions"],),
            "group": "agents",
            "dtype": th.float,
        },
        "reward": {"vshape": (1,)},
        "agent_reward": {"vshape": (env_info["l_n_agents"],)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    l_groups = {"agents": args.l_n_agents}

    l_preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.l_n_actions)])}

    # 领航者智能体 buffer
    l_buffer = PrioritizedReplayBuffer(
        l_scheme,
        l_groups,
        l_args.buffer_size,
        env_info["episode_limit"] + 1,
        args.per_alpha,
        args.per_beta,
        args.t_max,
        preprocess=l_preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # 生成多智能体控制器（mac）
    l_mac = mac_REGISTRY[l_args.mac](l_buffer.scheme, l_groups, l_args)

    # 设置 runner
    runner.setup(
        l_scheme=l_scheme,
        l_group=l_groups,
        l_preprocess=l_preprocess,
        l_mac=l_mac,
    )

    # 生成学习器（learner）
    l_learner = le_REGISTRY[l_args.learner](l_mac, l_buffer.scheme, logger, l_args)

    if args.use_cuda:
        l_learner.cuda()

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
            # episode_begin_t = time.time()
            l_episode_batch = runner.run(test_mode=False)
            # episode_end_t = time.time()
            # print(f"Episode Time: {episode_end_t-episode_begin_t}")
            l_buffer.insert_episode_batch(l_episode_batch)

        # 检查经验缓存是否足够进行采样，如果足够，就进行训练
        if l_buffer.can_sample(l_args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            for _ in range(args.train_times):
                # 通过 buffer.sample(args.batch_size) 从缓存中抽样一个批次的数据
                l_episode_sample, l_idx, l_weights = l_buffer.sample(
                    l_args.batch_size, runner.t_env
                )

                max_ep_t = l_episode_sample.max_t_filled()
                l_episode_sample = l_episode_sample[:, :max_ep_t]

                if l_episode_sample.device != args.device:
                    l_episode_sample.to(args.device)

                # logger.console_logger.info("神经网络更新")
                l_info = l_learner.train(
                    l_episode_sample, runner.t_env, episode, l_weights
                )
                del l_episode_sample

                # update priorities
                l_new_priorities = l_info["td_errors_abs"].flatten() + 1e-6
                l_buffer.update_priorities(l_idx, l_new_priorities.numpy().tolist())

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0 or (
            runner.t_env == args.t_max
        ):
            logger.console_logger.info("开始测试")
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

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            l_learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
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
