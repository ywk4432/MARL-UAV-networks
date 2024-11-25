import datetime
import os
import pprint
import threading
import time
from os.path import dirname, abspath
from types import SimpleNamespace as SN

import numpy as np
import torch as th

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
# from utils.plot_func import loss_abs_plot
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):
    """
    负责实验的初始化、运行、清理等过程
    构建实验参数变量 args 以及一个自定义 Logger 类的记录器 logger
    _config 是字典变量，查看参数时，需要利用 _config[key]=value
    """

    # check args sanity 检查参数是否正常
    _config = args_sanity_check(_config, _log)

    # 创建一个包含配置参数的命名空间对象 args
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info(f"主进程ID：{os.getpid()}, Two Sample")
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
    run_sequential(args=args, logger=logger)

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


def run_sequential(args, logger):
    """实验运行的主要函数"""

    # 生成运行器（runner）
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # 获取环境信息 env_info，包括智能体数量、动作数量、状态形状等
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)
    args.os_pid = os.getpid()

    if getattr(args, "agent_own_state_size", False):
        print("智能体无自己的观测尺寸")

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.float,
        },
        "reward": {"vshape": (1,)},
        "agent_reward": {"vshape": (env_info["n_agents"],)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # 生成 buffer
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # 生成高reward buffer
    buffer_reward = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # 生成多智能体控制器（mac）
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # 设置 runner
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 生成学习器（learner）
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # 遍历指定路径 args.checkpoint_path 下的所有文件和文件夹，然后筛选出满足一定条件的文件夹，将这些文件夹的名称（假设是数字）添加到列表 timesteps 中
        # os.listdir(args.checkpoint_path): 获取指定路径下的所有文件和文件夹的名称列表
        for name in os.listdir(args.checkpoint_path):
            #  构造完整的路径，将 args.checkpoint_path 和当前文件或文件夹的名称拼接在一起，形成完整的路径。
            full_name = os.path.join(args.checkpoint_path, name)
            # 检查当前路径是否是一个文件夹，检查当前名称是否由数字组成
            if os.path.isdir(full_name) and name.isdigit():
                # 当前路径是一个数字命名的文件夹，将这个数字转换为整数，并添加到列表 timesteps 中
                timesteps.append(int(name))

        # 确定要加载的模型的时间步（timestep）
        if args.load_step == 0:
            # args.load_step 的值为 0，表示要加载最新的模型。此时，选择 timesteps 列表中的最大值，即选择具有最大时间步的模型
            timestep_to_load = max(timesteps)
        else:
            # args.load_step 的值不为 0，表示要加载特定时间步的模型。此时，选择 timesteps 列表中与 args.load_step 最接近的时间步的模型
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        # 获取待加载模型路径
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
            # 通过 runner.run(test_mode=False) 运行了整个 episode，并将结果存储在 episode_batch 中
            episode_batch = runner.run(test_mode=False)
            # 使用 buffer.insert_episode_batch(episode_batch) 将 episode_batch 插入到经验缓存中
            buffer.insert_episode_batch(episode_batch)

        episode_batch_mean_reward = episode_batch["reward"].mean()
        if episode_batch_mean_reward > 0:
            buffer_reward.insert_episode_batch(episode_batch)

        # 检查经验缓存是否足够进行采样，如果足够，就进行训练
        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            if (
                args.is_two_sample
                and buffer_reward.can_sample(args.batch_size)
                and (np.random.rand() > args.highreward_sample)
            ):
                # 从高reward中采样
                episode_sample = buffer_reward.sample(args.batch_size)

            else:
                # 通过 buffer.sample(args.batch_size) 从缓存中抽样一个批次的数据
                episode_sample = buffer.sample(args.batch_size)

            # 将批次数据进行处理，确保只使用填充数据的有效时间步，通过 episode_sample = episode_sample[:, :max_ep_t] 进行截断
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 将批次数据传递给 learner.train 进行训练
            # logger.console_logger.info("神经网络更新")
            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        # 计算要执行的测试运行的次数。这个计算确保至少运行一次测试，即使 args.test_nepisode 很小。
        # args.test_nepisode 表示总的测试集数，runner.batch_size 表示每个测试运行中的批次大小。
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # 检查是否需要执行测试运行。在特定的时间间隔（由 args.test_interval 控制），通过 runner.run(test_mode=True) 进行测试运行。这将运行测试模式下的一个 episode。
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
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
            learner.save_models(save_path)

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
