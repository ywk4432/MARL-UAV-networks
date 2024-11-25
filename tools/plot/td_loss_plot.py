import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data(
    file_paths: list,
    window_size: int,
    window_step: int,
    truncate: float,
    episode_count: int,
) -> list:
    res = []
    for path in file_paths:
        with open(Path(path) / "td_error_abs_agent.csv", "r") as f:
            data = f.readlines()
        data = list(map(float, data))
        if truncate < 1:
            data = data[: int(len(data) * truncate)]
        data_min = data.copy()
        data_max = data.copy()
        data = [
            np.mean(data[i : min(i + window_size, len(data))])
            for i in range(0, len(data), window_step)
        ]
        data_min = [
            np.min(data_min[i : min(i + window_size, len(data_min))])
            for i in range(0, len(data_min), window_step)
        ]
        data_max = [
            np.max(data_max[i : min(i + window_size, len(data_max))])
            for i in range(0, len(data_max), window_step)
        ]
        res.append(
            [
                np.linspace(0, episode_count * truncate / 1e4, num=len(data)),
                data,
                data_min,
                data_max,
            ]
        )
    return res


def plot(save_path: Path, plot_data: list, labels: list, fig_name: str) -> None:
    """
    Args:
        save_path: 保存图片的路径
        plot_data: 绘图需要的数据
        labels: 对应数据的标签
        fig_name: 保存图片的文件路径
    """
    plt.figure(figsize=(6, 4), dpi=200)
    plt.xlabel("Training Episode ($10^4$)", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(axis="y")
    colors = ["red", "green", "blue", "cyan", "magenta"]
    for i in range(len(plot_data)):
        plt.plot(
            plot_data[i][0],
            plot_data[i][1],
            color=colors[i],
            label=labels[i],
        )
        plt.fill_between(
            plot_data[i][0],
            plot_data[i][2],
            plot_data[i][3],
            alpha=0.15,
            color=colors[i],
        )
    plt.legend(fontsize=12)
    if fig_name is None:
        fig_name = datetime.now().strftime("%m_%d_%H_%M_%S")
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.savefig(save_path / "png" / f"{fig_name}.png", bbox_inches="tight")
    plt.savefig(save_path / "pdf" / f"{fig_name}.pdf", bbox_inches="tight")
    plt.close()


def main(
    file_paths: list,
    window_step: int,
    labels: list,
    window_size: int,
    truncate: float,
    fig_name: str,
    episode_count: int,
) -> None:
    data = get_data(file_paths, window_size, window_step, truncate, episode_count)
    save_path = Path("fig/TD_error_abs")
    plot(save_path, data, labels, fig_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files_csv", type=str, default=None)
    parser.add_argument("-S", "--window-step", type=int, default=1)
    parser.add_argument("-s", "--window-size", type=int, default=20)
    parser.add_argument("-t", "--truncate", type=float, default=1)
    parser.add_argument("-o", "--fig-name", type=str, default=None)
    parser.add_argument("-e", "--episode-count", type=int, default=None)
    args = parser.parse_args()
    files_csv = pd.read_csv(args.files_csv, encoding="utf-8")
    args.loss_file_path = files_csv["file_path"].tolist()
    args.labels = files_csv["label"].tolist()
    assert len(args.loss_file_path) == len(args.labels)
    main(
        args.loss_file_path,
        args.window_step,
        args.labels,
        args.window_size,
        args.truncate,
        args.fig_name,
        args.episode_count,
    )
