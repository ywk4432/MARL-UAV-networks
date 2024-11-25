import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

dense = 1  # 格子内超过 1 人的为红色，否则为蓝色


def get_pos(
    config: dict,
    ue_file_path: Path,
    uav_file_path: list,
    uav_init_pos_file: Path,
    obs_list_file: str,
):
    """
    从文件中读取 UE 和 UAV 的坐标
    :param config: 环境配置文件字典
    :param ue_file_path: 存储 UE 位置信息的文件路径
    :param uav_file_path: 存储 UAV 所有时隙位置信息的文件路径 [path]
    :param uav_init_pos_file: 存储 UAV 初始位置的文件路径
    :param obs_list_file: 存储障碍物位置信息的文件路径
    :return: 供 plot 函数使用的参数 ue_pos, uav_pos, uav_radius
    """
    ue_pos_file = pd.read_csv(ue_file_path, usecols=["x", "y"]).values
    offset = np.random.uniform(low=0, high=0.5, size=ue_pos_file.shape)
    ue_pos_file = ue_pos_file.astype(np.float64) + offset
    ue_pos = [ue_pos_file[:, 0], ue_pos_file[:, 1], np.zeros(len(ue_pos_file))]
    uav_pos_file = [
        pd.read_csv(path, usecols=["pos"]).values.reshape(-1).tolist()
        for path in uav_file_path
    ]
    uav_init_pos = pd.read_csv(uav_init_pos_file, usecols=["init_pos"]).values.tolist()
    for i, uav_data in enumerate(uav_pos_file):
        uav_data = uav_init_pos[i] + uav_data
        for j in range(len(uav_data)):
            uav_data[j] = eval(uav_data[j])
        uav_pos_file[i] = np.array(uav_data, dtype=np.int32) + 0.5
        uav_init_pos[i] = uav_data[0]
    uav_pos = [[file[:, 0], file[:, 1], file[:, 2]] for file in uav_pos_file]
    uav_init_radius = [
        config["uav"]["cover_radius"][uav_init_pos[i][2]]
        for i in range(len(uav_pos_file))
    ]
    uav_radius = [
        [uav_init_radius[i]]
        + pd.read_csv(path, usecols=["cover_radius"]).values.reshape(-1).tolist()
        for i, path in enumerate(uav_file_path)
    ]
    obs_list = pd.read_csv(obs_list_file).values if obs_list_file is not None else []
    return ue_pos, uav_pos, uav_radius, obs_list


def plot_obstacles(obs_list: list, ax: plt.Axes):
    for obs in obs_list:
        x, y, z = obs[1], obs[2], 0
        a, b, c = obs[3], obs[4], obs[5]
        real = [
            [x + a, y, z],
            [x + a, y, z + c],
            [x + a, y + b, z + c],
            [x + a, y + b, z],
            [x + a, y, z],
            [x, y, z],
            [x, y, z + c],
            [x + a, y, z + c],
            [x + a, y + b, z + c],
            [x, y + b, z + c],
            [x, y, z + c],
        ]
        real = np.array(real).transpose()

        image1 = [
            [x, y, z],
            [x, y + b, z],
            [x, y + b, z + c],
        ]
        image1 = np.array(image1).transpose()
        image2 = [
            [x, y + b, z],
            [x + a, y + b, z],
        ]
        image2 = np.array(image2).transpose()

        ax.plot(real[0], real[1], real[2], c="k", linestyle="-")
        ax.plot(image1[0], image1[1], image1[2], c="k", linestyle="dashed")
        ax.plot(image2[0], image2[1], image2[2], c="k", linestyle="dashed")


def plot_static(obs_list: list, map_shape: list) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制地图的静态信息
    """
    fig = plt.figure(figsize=(15, 15), dpi=200)
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0, map_shape[0])
    ax.set_ylim(0, map_shape[1])
    ax.set_zlim(0, map_shape[2])
    plot_obstacles(obs_list, ax)
    return fig, ax


def episode_plot(
    ue_pos: list,
    uav_pos: list,
    uav_radius: list,
    obs_list: list,
    map_shape: list,
):
    """
    绘制 UAV 的轨迹，UE 的位置和障碍物的位置、形状
    :param ue_pos: 类型为 [np.array, np.array]， 表示所有 UE 的 x, y 坐标
    :param uav_pos: 列表，元素类型为 [np.array, np.array, np.array]， 表示某一个 UAV 在所有时隙的 x, y, z 坐标
    :param uav_radius: 列表，元素类型为 np.array，表示某一个 UAV 在所有时隙的覆盖半径
    :param obs_list: 列表, 为直接从 obs_list.csv 读取得到
    :param map_shape: 地图形状
    """
    fig, ax = plot_static(obs_list, map_shape)
    # 绘制 UE
    ue_cnt = np.zeros((map_shape[0], map_shape[1]), dtype=np.int32)
    for i in range(len(ue_pos[0])):
        ue_cnt[int(ue_pos[0][i]), int(ue_pos[1][i])] += 1
    ue_color = [
        "red" if ue_cnt[int(ue_pos[0][i]), int(ue_pos[1][i])] > dense else "gray"
        for i in range(len(ue_pos[0]))
    ]
    ax.scatter(ue_pos[0], ue_pos[1], ue_pos[2], color=ue_color, s=20, label="UE")
    # 绘制 UAV
    markers = ["o", "^", "s", "p", "*", "v", "<", ">"]
    colors = ["green", "red", "blue", "cyan", "magenta", "deepskyblue", "black", "gray"]
    for uav_index in range(len(uav_pos)):
        x, y, z = uav_pos[uav_index]
        radius = uav_radius[uav_index]
        # 绘制 UAV 轨迹
        ax.plot(x, y, z, c=colors[uav_index], linestyle="-")
        # 绘制平面投影
        ax.plot(x, y, c=colors[uav_index], linestyle="dashed")
        # 绘制 UAV 位置标记
        ax.scatter(
            x,
            y,
            z,
            c=colors[uav_index],
            marker=markers[uav_index],
            s=50,
            label=f"UAV {uav_index}",
        )
        # 绘制无人机在最后时隙的覆盖范围
        slot_cnt = len(x)
        theta = np.arange(0, 2 * np.pi, 0.05)
        a = x[slot_cnt - 1] + (radius[slot_cnt - 1] + 0.5) * np.cos(theta)
        b = y[slot_cnt - 1] + (radius[slot_cnt - 1] + 0.5) * np.sin(theta)
        real_a = []
        real_b = []
        for j in range(len(a)):
            if 0 < a[j] < map_shape[0] and 0 < b[j] < map_shape[1]:
                real_a.append(a[j])
                real_b.append(b[j])
        ax.plot(real_a, real_b, color=colors[uav_index], linestyle=":")
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.9), fontsize=20)
    return fig, ax


def plot(config: dict, prefix: Path, title: str, output_path: Path, max_height: int):
    uav_file_prefix = prefix / "uav"
    uav_init_pos_file = prefix / "uav" / "init_pos.csv"
    uav_file_path = sorted(uav_file_prefix.glob("uav_*.csv"))
    obs_file = config["obs_list_file"] if "obs_list_file" in config else None
    ue_pos, uav_pos, uav_radius, obs_list = get_pos(
        config, config["ue_pos_file"], uav_file_path, uav_init_pos_file, obs_file
    )
    map_shape = [config["map_length"], config["map_width"], config["map_height"]]
    if max_height is not None:
        map_shape[2] = max_height
    fig, ax = episode_plot(ue_pos, uav_pos, uav_radius, obs_list, map_shape)
    png_path = output_path / "png"
    pdf_path = output_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    fig.savefig(png_path / f"{title}.png", bbox_inches="tight")
    fig.savefig(pdf_path / f"{title}.pdf", bbox_inches="tight")


def batch_plot(config: dict, run_id: str, title_csv: Path, max_height: int) -> None:
    """
    批量绘制无人机飞行轨迹图
    Args:
        config: 环境配置字典
        run_id: 运行 ID
        title_csv: 存储绘图的标题的 CSV 文件路径
        max_height: 绘图时地图的最大高度
    """
    record_path = Path("record") / run_id
    titles = pd.read_csv(title_csv)
    for _, row in titles.iterrows():
        paths = record_path / str(row["pid"])
        if len(list(paths.glob("system_status.csv"))) == 0:
            prefix = sorted(map(lambda x: int(x.name), paths.iterdir()))[-1]
            prefix = record_path / f"{row['pid']}/{prefix}"
        else:
            prefix = paths
        title = row["title"]
        output_path = Path(f"fig/episode_path/{run_id}")
        plot(config, prefix, title, output_path, max_height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("run_id_or_prefix", type=str)
    parser.add_argument("-t", "--title", type=str, default=None)
    parser.add_argument("-o", "--output-path", type=str, default=None)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--max-height", type=int, default=None)
    args = parser.parse_args()
    with open(f"src/config/envs/{args.config}.yaml", "r", encoding="utf-8") as f:
        config: dict = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    if args.single:
        prefix = args.run_id_or_prefix
        if args.output_path is None:
            args.output_path = Path(f"fig/episode_path/{prefix})")
        else:
            args.output_path = Path(args.output_path)
        plot(config, Path(prefix), args.title, args.output_path, args.max_height)
    else:
        run_id = args.run_id_or_prefix
        if args.title is None:
            args.title = f"record/{run_id}/titles.csv"
        batch_plot(config, run_id, Path(args.title), args.max_height)


if __name__ == "__main__":
    main()
