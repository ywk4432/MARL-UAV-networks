'''
Date: 2023-11-25 09:08:40
description: 主要定义了一个 Logger 类和一个用于获取 logger 的函数 get_logger()
LastEditors: Wenke Yuan
LastEditTime: 2023-11-25 13:00:45
FilePath: /pymarl2-master/src/utils/logging.py
'''


from collections import defaultdict

import torch as th

import logging


class Logger:
    def __init__(self, console_logger):
        """接受一个 console_logger 作为参数，通常是通过 get_logger() 函数获得的

        Args:
            console_logger (_type_): _description_
        """
        self.console_logger = console_logger

        # 初始化了一些标志位，表示是否使用 TensorBoard (use_tb)、Sacred (use_sacred) 和 HDF (use_hdf)
        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        """
        - 创建 defaultdict 用于存储统计信息
        - defaultdict 是 Python 中的一个字典子类，它允许设置默认值，即在访问字典中不存在的键时，可以指定一个默认值。
        在这里，defaultdict(lambda: []) 表示如果访问字典中不存在的键，将返回一个空列表作为默认值。
        - 这种设计的好处是，可以方便地向字典中的每个键对应的列表中添加新的统计信息，而不必事先为每个键创建空列表。
        默认值为 [] 保证了如果某个键在 self.stats 中不存在，将会被初始化为空列表。
        这样的结构使得在实验过程中动态记录和更新各种统计信息变得简单和灵活。
        """
        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        """用于设置 TensorBoard 日志记录。接受一个目录名 directory_name 作为参数，配置 TensorBoard 记录到该目录。

        Args:
            directory_name (_type_): _description_
        """
        # Import here so it doesn't have to be installed if you don't use it
        # 导入 tensorboard_logger 模块，配置 TensorBoard，并将 log_value 函数绑定到 self.tb_logger
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        # 设置了 use_tb 标志位为 True
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        """用于设置 Sacred 日志记录。接受一个 Sacred 运行的信息字典 sacred_run_dict 作为参数

        Args:
            sacred_run_dict (_type_): _description_
        """
        # 将 sacred_run_dict.info 绑定到 self.sacred_info
        self.sacred_info = sacred_run_dict.info
        # 设置了 use_sacred 标志位为 True
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        """记录统计信息，包括键值 key、数值 value 和时间 t

        Args:
            key (_type_): _description_
            value (_type_): _description_
            t (_type_): _description_
            to_sacred (bool, optional): _description_. Defaults to True.
        """

        # 将记录保存到 self.stats 中
        self.stats[key].append((t, value))

        # 如果启用了 TensorBoard (use_tb)，调用 self.tb_logger 记录到 TensorBoard
        if self.use_tb:
            self.tb_logger(key, value, t)

        # 如果启用了 Sacred (use_sacred)，并且 to_sacred 参数为 True，将信息记录到 self.sacred_info 中
        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        """打印最近的统计信息到控制台和日志中。对于每个统计信息，计算最近几个值的平均，并按格式打印。
        """

        # 打印时间 t_env 和 Episode 的最新值
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1])

        """
        1. 初始化变量 i：
            i 被初始化为 0，用于记录当前处理的统计信息的数量。
        2. 迭代 self.stats 字典的键值对：
            通过 sorted(self.stats.items()) 对字典的键值对进行排序，以确保输出的顺序是按照键的字母顺序排列的。
        3. 跳过特定的键：
            在迭代中，如果键 k 是 "episode"，则通过 continue 跳过，不处理这个特定的键。
        4. 计算窗口大小 window：
            根据当前的键 k，确定窗口大小 window。如果键不是 "epsilon"，则窗口大小为 5，否则为 1。
        5. 计算 item：
            根据最近 window 个数据点计算平均值，并格式化为包含四位小数的字符串。
        6. 构建日志字符串：
            使用 "{:<25}{:>8}".format(k + ":", item) 格式化字符串，将键值对应的统计信息添加到 log_str 中。
        7. 处理换行符：
            检查 i 是否是 4 的倍数，如果是，添加换行符；否则，添加制表符。
        """
        # print("=== * 日志打印处 * ===")
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(
                th.mean(th.tensor([float(x[1]) for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        # 将构建好的最近统计信息的字符串 log_str 记录到日志中，使用的日志级别是 INFO。
        # self.console_logger 是一个用于在控制台输出日志信息的 Logger 对象。
        # 通过 info(log_str) 方法，将 log_str 中包含的最近统计信息以 INFO 级别输出到日志中。可以在控制台或日志文件中查看实验的运行状态和相关统计信息。
        self.console_logger.info(log_str)
        # 重置了 self.stats 以避免在内存中积累日志
        self.stats = defaultdict(lambda: [])


# set up a custom logger
def get_logger():
    """用于设置自定义日志记录器（logger）的函数

    Returns:
        _type_: _description_
    """
    # 获取一个日志记录器对象。Python 内置的 logging 模块提供的函数，用于创建或获取一个全局的日志记录器。
    logger = logging.getLogger()
    # 清除已存在的日志处理器。确保在设置新的处理器之前，不会有任何已存在的处理器。
    logger.handlers = []
    # 创建一个处理器（handler），这里是一个流处理器（StreamHandler），用于将日志消息输出到控制台。
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 创建一个格式化器（formatter），规定了日志输出的格式。
    # 在这里，日志消息的格式包括日志级别、时间、记录器名字和具体消息。'%H:%M:%S' 是时间的格式，表示小时:分钟:秒。
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    # 将格式化器应用到处理器上，确保输出的日志信息按照指定的格式进行排列。
    ch.setFormatter(formatter)
    # 将处理器添加到日志记录器中。这样，日志记录器就知道将日志消息发送到哪里，这里是发送到控制台。
    logger.addHandler(ch)
    # 设置日志记录器的日志级别为 DEBUG。这表示记录器将处理所有级别的日志消息。
    # 如果不设置，默认的级别是 WARNING，表示只处理 WARNING 及以上级别的消息。
    logger.setLevel('DEBUG')

    return logger
