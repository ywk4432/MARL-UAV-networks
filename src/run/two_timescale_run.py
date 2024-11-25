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
    """
    负责实验的初始化、运行、清理等过程
    构建实验参数变量 args 以及一个自定义 Logger 类的记录器 logger
    _config 是字典变量，查看参数时，需要利用 _config[key]=value
    """

    # check args sanity 检查参数是否正常
    _config = args_sanity_check(_config, _log)
    _l_config = copy.deepcopy(_config)
    _f_config = copy.deepcopy(_config)

    for k in _config["l_args"]:
        _l_config[k] = _config["l_args"][k]
    for k in _config["f_args"]:
        _f_config[k] = _config["f_args"][k]

    # 创建一个包含配置参数的命名空间对象 args
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    l_args = SN(**_l_config)
    l_args.device = "cuda" if l_args.use_cuda else "cpu"
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
    run_sequential(args=args, logger=logger, l_args=l_args, f_args=f_args)

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


def run_sequential(args, logger, l_args, f_args):
    """实验运行的主要函数"""

    # 生成运行器（runner）
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # 获取环境信息 env_info，包括智能体数量、动作数量、状态形状等
    env_info = runner.get_env_info()
    args.l_n_agents = env_info["l_n_agents"]
    args.f_n_agents = env_info["f_n_agents"]
    args.l_n_actions = env_info["l_n_actions"]
    args.f_n_actions = env_info["f_n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    args.os_pid = os.getpid()

    # 设置领航无人机参数信息
    l_args.n_agents = env_info["l_n_agents"]
    l_args.n_actions = env_info["l_n_actions"]
    l_args.state_shape = env_info["state_shape"]
    l_args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    l_args.os_pid = os.getpid()
    # 设置跟随无人机参数信息
    f_args.n_agents = env_info["f_n_agents"]
    f_args.n_actions = env_info["f_n_actions"]
    f_args.state_shape = env_info["state_shape"]
    f_args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    f_args.os_pid = os.getpid()

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

    l_preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.l_n_actions)])}
    f_preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.f_n_actions)])}

    # 领航者智能体 buffer
    l_buffer = ReplayBuffer(
        l_scheme,
        l_groups,
        l_args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=l_preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
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
    l_mac = mac_REGISTRY[l_args.mac](l_buffer.scheme, l_groups, l_args)
    f_mac = mac_REGISTRY[f_args.mac](f_buffer.scheme, f_groups, f_args)

    # 设置 runner
    runner.setup(
        l_scheme=l_scheme,
        f_scheme=f_scheme,
        l_group=l_groups,
        f_group=f_groups,
        l_preprocess=l_preprocess,
        f_preprocess=f_preprocess,
        l_mac=l_mac,
        f_mac=f_mac,
    )

    # 生成学习器（learner）
    l_learner = le_REGISTRY[l_args.learner](l_mac, l_buffer.scheme, logger, l_args)
    f_learner = le_REGISTRY[f_args.learner](f_mac, f_buffer.scheme, logger, f_args)

    if args.use_cuda:
        l_learner.cuda()
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
        # Run for a whole episode at a time
        with th.no_grad():
            # 通过 runner.run(test_mode=False) 运行了整个 episode，并将结果存储在 episode_batch 中
            # episode_begin_t = time.time()
            l_episode_batch, f_episode_batch = runner.run(test_mode=False)
            # episode_end_t = time.time()
            # print(f"Episode Time: {episode_end_t-episode_begin_t}")
            # 使用 buffer.insert_episode_batch(episode_batch) 将 episode_batch 插入到经验缓存中
            l_buffer.insert_episode_batch(l_episode_batch)
            f_buffer.insert_episode_batch(f_episode_batch)

        # 检查经验缓存是否足够进行采样，如果足够，就进行训练
        if l_buffer.can_sample(l_args.batch_size) and f_buffer.can_sample(
            f_args.batch_size
        ):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            for _ in range(args.train_times):
                # 通过 buffer.sample(args.batch_size) 从缓存中抽样一个批次的数据
                l_episode_sample = l_buffer.sample(l_args.batch_size)
                f_episode_sample = f_buffer.sample(f_args.batch_size)

                # 将批次数据进行处理，确保只使用填充数据的有效时间步，通过 episode_sample = episode_sample[:, :max_ep_t] 进行截断
                max_ep_t = l_episode_sample.max_t_filled()
                l_episode_sample = l_episode_sample[:, :max_ep_t]

                if l_episode_sample.device != args.device:
                    l_episode_sample.to(args.device)

                max_ep_t = f_episode_sample.max_t_filled()
                f_episode_sample = f_episode_sample[:, :max_ep_t]

                if f_episode_sample.device != args.device:
                    f_episode_sample.to(args.device)

                # 将批次数据传递给 learner.train 进行训练
                # logger.console_logger.info("神经网络更新")
                l_learner.train(l_episode_sample, runner.t_env, episode)
                f_learner.train(f_episode_sample, runner.t_env, episode)
                del l_episode_sample, f_episode_sample

        # Execute test runs once in a while
        # 计算要执行的测试运行的次数。这个计算确保至少运行一次测试，即使 args.test_nepisode 很小。
        # args.test_nepisode 表示总的测试集数，runner.batch_size 表示每个测试运行中的批次大小。
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # 检查是否需要执行测试运行。在特定的时间间隔（由 args.test_interval 控制），通过 runner.run(test_mode=True) 进行测试运行。这将运行测试模式下的一个 episode。
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

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            l_learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    """
    检查配置参数的合法性，并根据需要进行一些自动调整
    接收两个参数，config 是配置参数的字典，_log 是用于记录日志的对象。
    """

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    # 如果 use_cuda 为 True 但 CUDA 不可用，将 use_cuda 设置为 False，并记录一条警告日志。
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    # 确保 test_nepisode 的值是 batch_size_run 的整数倍。
    # 如果小于 batch_size_run，将其设为 batch_size_run；否则，将其调整为最接近但小于原值的 batch_size_run 的整数倍。
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
