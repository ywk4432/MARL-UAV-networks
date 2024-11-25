import argparse
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def read_data(
    ue_file_path: Union[str, Path],
    uav_file_path: list,
    uav_init_pos_file: Union[str, Path],
    obs_list_file: Union[str, Path],
    truncate: float,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    从文件中读取保存的数据
    :param ue_file_path: 存储 UE 位置信息的文件路径
    :param uav_file_path: 存储 UAV 所有时隙位置信息的文件路径 [path]
    :param uav_init_pos_file: 存储 UAV 初始位置的文件路径
    :param obs_list_file: 存储障碍物位置信息的文件路径
    :param truncate: 只绘制所有时隙中前 truncate 比例的部分
    :return: [ue: [x, y]]: np.ndarray,
             [uav: [slot: [x, y, z]]]: list(np.ndarray),
             [obs: [x, y, a, b]] np.ndarray
    """
    ue_pos = pd.read_csv(ue_file_path, usecols=["x", "y"]).values
    offset = np.random.uniform(low=0, high=0.5, size=ue_pos.shape)
    ue_pos = ue_pos.astype(np.float64) + offset
    uav_data = [pd.read_csv(path) for path in uav_file_path]
    uav_final_cover_radius = []
    uav_pos = []
    uav_init_pos = pd.read_csv(uav_init_pos_file)["init_pos"]
    for i, data in enumerate(uav_data):
        pos = list(map(eval, [uav_init_pos[i]] + data["pos"].tolist()))
        pos = np.array(pos, dtype=np.float64) + 0.5
        uav_pos.append(pos[: round(truncate * len(pos))])
        uav_final_cover_radius.append(data["cover_radius"].iloc[-1])
    obs_list = (
        pd.read_csv(obs_list_file, usecols=["pos_x", "pos_y", "length", "width"]).values
        if obs_list_file is not None
        else []
    )
    return ue_pos, uav_pos, obs_list


def plot_static(ue_pos: np.ndarray, obs_list: np.ndarray):
    plt.scatter(ue_pos[:, 0], ue_pos[:, 1], c="gray", s=10, label="UE")
    for x, y, a, b in obs_list:
        points = [[x, y], [x + a, y], [x + a, y + b], [x, y + b], [x, y]]
        points = np.array(points).transpose()
        plt.fill(points[0], points[1], c="black", linestyle="-")


def plot_uav(uav_pos: List[np.ndarray]):
    colors = ["green", "red", "blue", "cyan", "magenta", "deepskyblue"]
    markers = ["o", "^", "s", "p", "*", "v"]
    for i, pos in enumerate(uav_pos):
        # 绘制 UAV 轨迹
        plt.plot(pos[:, 0], pos[:, 1], c=colors[i], linestyle="-")
        # 绘制 UAV 位置标记
        plt.scatter(
            pos[:, 0],
            pos[:, 1],
            c=colors[i],
            marker=markers[i],
            s=20,
            label=f"UAV {i}",
        )


def plot_novelty(config: dict, uav_pos: List[np.ndarray]):
    colors = ["green", "red", "blue", "cyan", "magenta", "deepskyblue"]
    map_length = config["map_length"]
    map_width = config["map_width"]
    for uav_id, pos_data in enumerate(uav_pos):
        for x, y, z in pos_data:
            x, y, z = x - 0.5, y - 0.5, z - 0.5
            radius = (
                config["uav"]["obs_radius"][z]
                if config["uav"]["obs_radius_change"]
                else config["uav"]["max_obs_radius"]
            )
            x_min, x_max = round(max(0, x - radius)), round(min(x + radius, map_length))
            y_min, y_max = round(max(0, y - radius)), round(min(y + radius, map_width))
            points = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
            points = np.array(points).transpose()
            plt.fill(points[0], points[1], color=colors[uav_id], alpha=0.2)


def plot(config: dict, prefix: Path, title: str, output_path: Path, truncate: float):
    """
    绘制投影图
    Args:
        config: 环境配置字典
        prefix: 环境 record 目录
        title: 图的标题
        output_path: 图的保存路径
        truncate: 只绘制所有时隙中前 truncate 比例的部分

    Returns:

    """
    plt.figure(dpi=200)
    plt.grid()
    plt.xlim(0, config["map_length"])
    plt.ylim(0, config["map_width"])
    uav_file_prefix = prefix / "uav"
    uav_init_pos_file = prefix / "uav" / "init_pos.csv"
    uav_file_path = sorted(uav_file_prefix.glob("uav_*.csv"))
    obs_file = config["obs_list_file"] if "obs_list_file" in config else None
    ue_pos, uav_pos, obs_list = read_data(
        config["ue_pos_file"], uav_file_path, uav_init_pos_file, obs_file, truncate
    )
    plot_novelty(config, uav_pos)
    plot_static(ue_pos, obs_list)
    plot_uav(uav_pos)
    plt.legend(loc="upper left")
    png_path = output_path / "png"
    pdf_path = output_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.savefig(png_path / f"{title}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{title}.pdf", bbox_inches="tight")
    plt.close()


def batch_plot(config: dict, run_id: str, title_csv: Path, truncate: float) -> None:
    """
    批量绘制无人机飞行轨迹图
    Args:
        config: 环境配置字典
        run_id: 运行 ID
        title_csv: 存储绘图的标题的 CSV 文件路径
        truncate: 只绘制所有时隙中前 truncate 比例的部分
    """
    record_path = Path("record") / run_id
    titles = pd.read_csv(title_csv)
    for _, row in titles.iterrows():
        paths = sorted((record_path / str(row["pid"])).iterdir())[:-1]
        prefix = sorted(map(lambda x: int(x.name), paths))[-1]
        prefix = record_path / f"{row['pid']}/{prefix}"
        title = row["title"]
        output_path = Path(f"fig/novelty/{run_id}")
        plot(config, prefix, title, output_path, truncate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("run_id_or_prefix", type=str)
    parser.add_argument("-t", "--title", type=str, default=None)
    parser.add_argument("-T", "--truncate", type=float, default=1)
    parser.add_argument("-o", "--output-path", type=str, default=None)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    with open(f"src/config/envs/{args.config}.yaml", "r", encoding="utf-8") as f:
        config: dict = yaml.load(f, Loader=yaml.FullLoader)["env_args"]
    if args.single:
        prefix = args.run_id_or_prefix
        if args.output_path is None:
            args.output_path = Path(f"fig/novelty/{prefix})")
        else:
            args.output_path = Path(args.output_path)
        plot(config, Path(prefix), args.title, args.output_path, args.truncate)
    else:
        run_id = args.run_id_or_prefix
        if args.title is None:
            args.title = f"record/{run_id}/titles.csv"
        batch_plot(config, run_id, Path(args.title), args.truncate)


if __name__ == "__main__":
    main()
