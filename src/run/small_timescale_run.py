import copy
import datetime
import os
import pprint
import threading
import time
from os.path import dirname, abspath
from types import SimpleNamespace as SN

import torch as th

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):

    # check args sanity 检查参数是否正常
    _config = args_sanity_check(_config, _log)
    _f_config = copy.deepcopy(_config)

    for k in _config["f_args"]:
        _f_config[k] = _config["f_args"][k]

    # 创建一个包含配置参数的命名空间对象 args
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    f_args = SN(**_f_config)
    f_args.device = "cuda" if f_args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info(f"主进程ID：{os.getpid()}, Default Sample")
    _log.info("Experiment Parameters:")
    # indent=4 表示每一层的缩进为 4 个空格，width=1 表示每行的宽度为 1，这样输出的效果是每个键值对都占用一行
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
    run_sequential(args=args, logger=logger, f_args=f_args)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            # join() 方法用于等待线程终止。timeout=1 表示最多等待 1 秒钟。
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


def run_sequential(args, logger, f_args):

    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.f_n_agents = env_info["f_n_agents"]
    args.f_n_actions = env_info["f_n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    args.os_pid = os.getpid()

    # 设置跟随无人机参数信息
    f_args.n_agents = env_info["f_n_agents"]
    f_args.n_actions = env_info["f_n_actions"]
    f_args.state_shape = env_info["state_shape"]
    f_args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    f_args.os_pid = os.getpid()

    if getattr(args, "agent_own_state_size", False):
        print("智能体无自己的观测尺寸")

    # Default/Base scheme
    f_scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["f_obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["f_n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "probs": {
            "vshape": (env_info["f_n_actions"],),
            "group": "agents",
            "dtype": th.float,
        },
        "reward": {"vshape": (1,)},
        "agent_reward": {"vshape": (env_info["f_n_agents"],)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    f_groups = {"agents": args.f_n_agents}

    f_preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.f_n_actions)])}

    # 跟随者智能体 buffer
    f_buffer = ReplayBuffer(
        f_scheme,
        f_groups,
        f_args.buffer_size,
        env_info["episode_limit"] * args.env_args["slot_step_num"] + 1,
        preprocess=f_preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # 生成多智能体控制器（mac）
    f_mac = mac_REGISTRY[f_args.mac](f_buffer.scheme, f_groups, f_args)

    # 设置 runner
    runner.setup(
        f_scheme=f_scheme,
        f_group=f_groups,
        f_preprocess=f_preprocess,
        f_mac=f_mac,
    )

    f_learner = le_REGISTRY[f_args.learner](f_mac, f_buffer.scheme, logger, f_args)

    if args.use_cuda:
        f_learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        with th.no_grad():
            f_episode_batch = runner.run(test_mode=False)
            f_buffer.insert_episode_batch(f_episode_batch)

        if f_buffer.can_sample(f_args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            for _ in range(args.train_times):
                f_episode_sample = f_buffer.sample(f_args.batch_size)

                max_ep_t = f_episode_sample.max_t_filled()
                f_episode_sample = f_episode_sample[:, :max_ep_t]

                if f_episode_sample.device != args.device:
                    f_episode_sample.to(args.device)

                f_learner.train(f_episode_sample, runner.t_env, episode)
                del f_episode_sample

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)

        if (runner.t_env - last_test_T) / args.test_interval >= 1.0 or (
            runner.t_env == args.t_max
        ):
            # 一开始就会进入这个分支
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

            # 第一次测试完之后，由于给 last_test_T 赋了 runner.t_env，下次就不会进来
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
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

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
