from functools import partial

from .env_1 import Env as env_ueCluster
from .env_2 import Env as env_Hotspot
from .env_3 import LUAVEnv as env_formation
from .env_4 import SNEnv as env_TPRA
from .env_5 import Env as env_multiObject
from .env_6 import Env as env_selectFormation
from .multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    """定义了一个工厂函数 (env_fn)，它接受一个环境类 (env) 和任意关键字参数 (**kwargs)。通过调用环境类的构造函数来创建并返回一个新的环境实例

    Args:
        env (_type_): _description_

    Returns:
        MultiAgentEnv: 使用了可变关键字参数 (**kwargs)，可以接受任意数量的关键字参数，并将它们传递给环境类的构造函数。这样的设计允许在创建环境实例时动态地传递配置参数。
    """
    return env(**kwargs)


REGISTRY = {}
REGISTRY["cluster"] = partial(env_fn, env=env_ueCluster)
REGISTRY["hotspot"] = partial(env_fn, env=env_Hotspot)
REGISTRY["formation"] = partial(env_fn, env=env_formation)
REGISTRY["SNode"] = partial(env_fn, env=env_TPRA)
REGISTRY["multiObject"] = partial(env_fn, env=env_multiObject)
REGISTRY["selectFormation"] = partial(env_fn, env=env_selectFormation)