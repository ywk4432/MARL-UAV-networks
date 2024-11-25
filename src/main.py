import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from os.path import dirname, abspath, join

import numpy as np
import torch as th
import yaml
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from run import REGISTRY as run_REGISTRY
from utils.logging import get_logger

# 设置 Sacred 运行参数
# set to "no" if you want to see stdout/stderr in console
SETTINGS["CAPTURE_MODE"] = "fd"
logger = get_logger()
ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds
results_path = join(dirname(dirname(abspath(__file__))), "results")

# cuda_index = 1
# th.cuda.set_device(cuda_index)


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    # 设置 NumPy 和 PyTorch 的随机数种子，以确保实验的可复现性。
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    if "obj" in config:
        config["env_args"]["obj"] = config["obj"]
    if "obj_type" in config:
        config["env_args"]["obj_type"] = config["obj_type"]

    # run
    run_REGISTRY[_config["run"]](_run, config, _log)


def _get_config(params, arg_name, subfolder):
    """
    从命令行参数中获取配置文件

    params (list()): 命令行传进来的参数列表
    arg_name : 参数名称
    subfolder : 配置文件所在的文件夹
    """

    """
    前半部分，python 中的 enumerate() 方法本质是一个生成器，返回有标号的序列： (0, params[0]), (1, params[1]) ... ，
    之后找赋值语句中有没有 --config/--env-config 这两个关键词，即参数 arg_name ：
    如果有，把指定的文件名称赋值给 config_name ，退出循环前把读取过的参数删了
    """
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    """
    后半部分按指定的 config_name.yaml 读取对应的配置文件内容，返回对应的字典型配置文件 config_dict ，
    这里操作和读取默认设置文件完全相同，只是算法和环境的文件存储在 algs/envs 子目录下
    """
    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
            encoding="utf-8",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    """
    递归地更新字典: 将字典 d 中的键值对按照递归的方式更新为字典 u 中对应的键值对
    """
    for k, v in u.items():
        # k 是键，v 是值
        if isinstance(v, Mapping):
            # isinstance 函数接受两个参数，第一个参数是待检查的对象，第二个参数是类型或类型的元组。它返回一个布尔值，表示对象是否是指定类型的实例
            d[k] = recursive_dict_update(d.get(k, {}), v)
            # d.get(k, {}) 的作用：如果字典 d 中存在键 k，则返回 d[k]；如果字典 d 中不存在键 k，则返回空字典 {}
        else:
            d[k] = v
    return d


def config_copy(config):
    """
    用于深拷贝配置
    """
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    """
    解析命令行参数
    """
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index("=") + 1 :].strip()
            break
    return result


if __name__ == "__main__":
    # 获取命令行参数
    params = deepcopy(sys.argv)

    # 从 default.yaml 中获取默认配置参数
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"),
        "r",
        encoding="utf-8",
    ) as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # 获取环境和算法两个参数配置，并更新配置字典
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}

    """
    有三种配置数据:
        默认设置 config_dict
        指定的算法设置 alg_config
        指定的环境设置 env_config 
    后两者优先级应当更高，用后面的设置覆盖默认设置中相同的条目：
    """
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(
        params, "env_args.map_name", config_dict["env_args"]["map_name"]
    )
    algo_name = parse_command(params, "name", config_dict["name"])
    file_obs_path = join(results_path, "sacred", map_name, algo_name)

    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 解析命令行参数并执行实验。在执行实验的过程中，Sacred 框架会自动调用 my_main 函数，并将相应的参数传递给该函数，以完成实验的具体逻辑。
    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
