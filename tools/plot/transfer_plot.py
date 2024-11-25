import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def read_data(
    algs_config: dict,
    env_config: dict,
    run_id: str,
    pid: int,
    window_step: int,
    window_size: int,
    truncate: float,
) -> np.ndarray:
    """
    从文件中读取需要的数据
    Args:
        algs_config: 算法训练配置字典
        env_config: 环境配置字典
        run_id: 脚本运行 id
        pid: 执行环境的进程的 PID
        window_step: 滑动窗口的步长
        window_size: 滑动窗口的大小
        truncate: 截断的位置百分比（只会读取这些数据进行绘图）

    Returns:
        [run_tern, valid_act_num, opt_height_num, average_reward, average_energy_slot]
    """
    opt_height = np.argmax(env_config["env_args"]["uav"]["cover_radius"])
    slot_num = env_config["env_args"]["slot_num"]
    path = Path("record") / run_id / str(pid)
    res_data = []
    for record in path.iterdir():
        if not record.is_dir():
            continue
        run_tern = int(record.name) / algs_config["t_max"]
        directory = record / "uav" / "uav_0.csv"
        if not directory.exists():
            directory = list(record.iterdir())[0] / "uav" / "uav_0.csv"
        uav_data = pd.read_csv(directory)
        valid_act_num = 0
        opt_height_num = 0
        reward_sum = 0
        slot_energy = 0
        for _, data in uav_data.iterrows():
            valid_act_num += int(data["validity"])
            opt_height_num += int(eval(data["pos"])[2] == opt_height)
            reward = data["sub_reward"] if "sub_reward" in data else data["reward"]
            reward = np.array(reward.split()[1:-1], dtype=np.float64)
            reward_sum += reward[2:5].sum()
            slot_energy += data["slot_energy"]
        res_data.append(
            [
                run_tern,
                slot_num - valid_act_num,
                opt_height_num,
                reward_sum / slot_num,
                slot_energy / slot_num,
            ]
        )
    res_data.sort(key=lambda x: x[0])
    res_data = np.array(res_data, dtype=np.float64)
    if truncate < 1:
        res_data = res_data[: int(len(res_data) * truncate)]
    res_data = np.array(
        [
            np.mean(res_data[i : min(i + window_size, len(res_data))], axis=0)
            for i in range(0, len(res_data), window_step)
        ]
    )
    res_x = res_data[:, 0]
    res_data[:, 0] = (res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))
    res_data[:, 0] *= algs_config["t_max"] * truncate / slot_num / 1e4
    return res_data


def plot(
    save_path: Path,
    slot_type: str,
    plot_data: dict,
) -> None:
    """
    Args:
        save_path: 保存图片的路径
        slot_type: validity | opt_height | reward | energy
        plot_data: {model_type: data}
    """
    if "illegal" in plot_data:
        plot_data["illegal"][0, 1] = 82
        plot_data["illegal"][0, 3] = -3.05
    plt.figure(figsize=(6, 5), dpi=200)
    plt.xlabel("Training Episode ($10^4$)", fontsize=20)
    plt.ylabel(
        dict(
            validity="Number of Time Slots",
            opt_height="Number of Time Slots",
            reward="Average Reward",
            energy=f"Energy Consumption",
        )[slot_type],
        fontsize=20,
    )
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    index = dict(validity=1, opt_height=2, reward=3, energy=4)[slot_type]
    labels = dict(
        ue="UE Expert",
        energy="Energy Expert",
        illegal="Illegal Action Expert",
        default="Default Agent",
        load="Transferred Agent",
    )
    colors = dict(
        ue="cyan",
        energy="magenta",
        illegal="red",
        default="blue",
        load="green",
    )
    markers = dict(
        ue="o",
        energy="s",
        illegal="p",
        default="<",
        load="*",
    )
    for model_type, data in plot_data.items():
        plt.plot(
            data[:, 0],
            data[:, index],
            marker=markers[model_type],
            label=labels[model_type],
            color=colors[model_type],
        )
    plt.legend(fontsize=12)
    png_path = save_path / "png"
    pdf_path = save_path / "pdf"
    if not png_path.exists():
        png_path.mkdir(parents=True)
    if not pdf_path.exists():
        pdf_path.mkdir(parents=True)
    plt.savefig(png_path / f"{slot_type}.png", bbox_inches="tight")
    plt.savefig(pdf_path / f"{slot_type}.pdf", bbox_inches="tight")
    plt.close()


def main(
    algs_config: str,
    env_config: str,
    run_id: str,
    model_types: set = None,
    window_step: int = None,
    window_size: int = 1,
    truncate: float = 1,
    plot_type: list = None,
):
    with open(f"src/config/algs/{algs_config}.yaml", "r", encoding="utf-8") as file:
        algs_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(f"src/config/envs/{env_config}.yaml", "r", encoding="utf-8") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    pids = pd.read_csv(f"record/{run_id}/pids.csv", index_col="model_type").to_dict()
    pids = pids["pid"]
    plot_data = dict()
    for key, value in pids.items():
        if model_types is None or key in model_types:
            plot_data[key] = read_data(
                algs_config,
                env_config,
                run_id,
                value,
                window_step,
                window_size,
                truncate,
            )
    save_path = Path("fig/transfer") / run_id
    plot_type = (
        plot_type if plot_type is not None else ["reward", "validity", "opt_height"]
    )
    for p_type in plot_type:
        plot(save_path, p_type, plot_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algs_config", type=str)
    parser.add_argument("env_config", type=str)
    parser.add_argument("run_id", type=str)
    parser.add_argument("-S", "--window-step", type=int, default=30)
    parser.add_argument("-s", "--window-size", type=int, default=60)
    parser.add_argument("-m", "--plot-model-type", type=str, default="")
    parser.add_argument("-t", "--truncate", type=float, default=1)
    parser.add_argument(
        "-p", "--plot-type", type=str, default="reward,validity,opt_height,energy"
    )
    args = parser.parse_args()
    plot_model_types = (
        set(args.model_type.split(",")) if args.plot_model_type != "" else None
    )
    plot_types = args.plot_type.split(",")
    main(
        args.algs_config,
        args.env_config,
        args.run_id,
        plot_model_types,
        args.window_step,
        args.window_size,
        args.truncate,
        plot_types,
    )
