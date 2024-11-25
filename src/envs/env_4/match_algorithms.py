import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from pathlib import Path




# 计算距离矩阵
def calculate_distance_matrix(uavs, clusters):
    uavs = np.array(uavs)
    clusters = np.array(clusters)
    
    n = len(uavs)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(uavs[i] - clusters[j])
    return distance_matrix


# 贪心算法实现
def greedy_matching(uavs, clusters):
    distance_matrix = calculate_distance_matrix(uavs, clusters)
    n = len(uavs)
    matching = [-1] * n  # 初始化匹配结果，-1表示未匹配
    cluster_chosen = [False] * n  # 初始化集群是否被选择的标记

    for i in range(n):
        # 获取当前无人机 i 到所有集群的距离，并按距离排序
        sorted_clusters = np.argsort(distance_matrix[i])
        for cluster_idx in sorted_clusters:
            # 选择最近的、未被选择的集群
            if not cluster_chosen[cluster_idx]:
                matching[i] = cluster_idx
                cluster_chosen[cluster_idx] = True
                break  # 退出循环，进入下一个无人机的匹配

    # 计算贪心算法的总代价
    total_cost = sum(distance_matrix[i, matching[i]] for i in range(n))
    return matching, total_cost


# 匈牙利算法实现
def hungarian_matching(uavs, clusters):
    distance_matrix = calculate_distance_matrix(uavs, clusters)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    total_cost = distance_matrix[row_ind, col_ind].sum()
    return col_ind, total_cost


# 排序匹配算法实现
def sorted_matching(uav_energy, cluster_data, uavs, clusters):
    distance_matrix = calculate_distance_matrix(uavs, clusters)
    # 按能量和数据量排序后，匹配索引
    uav_sorted_indices = np.argsort(uav_energy)
    cluster_sorted_indices = np.argsort(cluster_data)

    # 匹配能量最小的UAV到数据量最小的cluster
    matching = [-1] * len(uavs)
    for i in range(len(uavs)):
        uav_idx = uav_sorted_indices[i]
        cluster_idx = cluster_sorted_indices[i]
        matching[uav_idx] = cluster_idx

    # 计算排序匹配算法的总代价
    total_cost = sum(distance_matrix[i, matching[i]] for i in range(len(uavs)))
    return matching, total_cost





# 绘制无人机整体能耗对比柱状图并保存
def plot_total_energy_cost_comparison(total_costs, algorithm_names):
    plt.figure(figsize=(8, 6))
    plt.bar(algorithm_names, total_costs, color=["blue", "green", "red"])
    plt.xlabel("Algorithm", fontsize=axis_fontsize)
    plt.ylabel("Total Energy Cost", fontsize=axis_fontsize)
    plt.title(
        "Total Energy Cost Comparison Across Different Algorithms",
        fontsize=title_fontsize,
    )
    plt.grid(axis="y")
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)

    # 保存图像
    plt.savefig(save_path / "total_energy_cost_comparison.pdf", format="pdf")
    plt.close()


# 绘制cluster数据量柱状图，并叠加UAV能量匹配折线图并保存
def plot_cluster_data_with_uav_energy(
    cluster_data,
    uav_energy,
    greedy_matching_result,
    hungarian_matching_result,
    sorted_matching_result,
):
    # 根据每种算法的匹配结果反向映射到各 cluster 的 UAV 能量
    greedy_uav_energy = [
        uav_energy[np.where(np.array(greedy_matching_result) == i)[0][0]]
        for i in range(len(cluster_data))
    ]
    hungarian_uav_energy = [
        uav_energy[np.where(np.array(hungarian_matching_result) == i)[0][0]]
        for i in range(len(cluster_data))
    ]
    sorted_uav_energy = [
        uav_energy[np.where(np.array(sorted_matching_result) == i)[0][0]]
        for i in range(len(cluster_data))
    ]

    # 绘制cluster数据量柱状图，并叠加UAV能量匹配折线图
    x = np.arange(len(cluster_data))  # Cluster编号

    plt.figure(figsize=(4, 3))
    plt.bar(x, cluster_data, color="lightblue", label="Cluster Data")
    # 不同算法的UAV能量匹配折线图，使用不同线型
    plt.plot(
        x,
        greedy_uav_energy,
        marker="o",
        color="b",
        linestyle="-",
        linewidth=2,
        label="Greedy Algorithm - UAV Energy",
    )
    plt.plot(
        x,
        hungarian_uav_energy,
        marker="s",
        color="g",
        linestyle="--",
        linewidth=2,
        label="Hungarian Algorithm - UAV Energy",
    )
    plt.plot(
        x,
        sorted_uav_energy,
        marker="^",
        color="r",
        linestyle="-.",
        linewidth=2,
        label="Sorted Matching Algorithm - UAV Energy",
    )

    # 图形设置
    plt.xlabel("Cluster Index", fontsize=axis_fontsize)
    plt.ylabel("Value", fontsize=axis_fontsize)
    plt.title(
        "Cluster Data with Matched UAV Energy Across Algorithms",
        fontsize=title_fontsize,
    )
    plt.xticks(x, [f"Cluster {i+1}" for i in x], fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.legend(fontsize=axis_fontsize)
    plt.grid(axis="y")

    # 保存图像
    plt.savefig(save_path / "cluster_data_with_uav_energy.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    # 设置随机数种子以便于结果复现
    np.random.seed(42)

    # 设置无人机和集群的数量
    N = 5

    # 随机生成无人机和集群的位置和属性，范围在 [0, 10] 之间
    uavs = np.random.rand(N, 2) * 10  # 生成无人机的位置
    clusters = np.random.rand(N, 2) * 10  # 生成集群的位置
    uav_energy = np.random.rand(N) * 100  # 随机生成无人机的能量
    cluster_data = np.random.rand(N) * 100  # 随机生成集群的数据量

    # 运行三种算法
    greedy_matching_result, greedy_total_cost = greedy_matching(uavs, clusters)
    hungarian_matching_result, hungarian_total_cost = hungarian_matching(uavs, clusters)
    sorted_matching_result, sorted_total_cost = sorted_matching(
        uav_energy, cluster_data, uavs, clusters
    )

    # 设置图表字体和保存路径
    plt.rcParams["font.family"] = "Times New Roman"
    title_fontsize = 16
    axis_fontsize = 14
    save_path = Path.cwd() / "fig"
    save_path.mkdir(exist_ok=True)

    # 总能耗数据和算法名称
    algorithm_names = ["Energy Greedy", "Hungarian", "Resource Greedy"]
    total_costs = [greedy_total_cost, hungarian_total_cost, sorted_total_cost]

    # 调用绘图函数
    plot_total_energy_cost_comparison(total_costs, algorithm_names)
    plot_cluster_data_with_uav_energy(
        cluster_data,
        uav_energy,
        greedy_matching_result,
        hungarian_matching_result,
        sorted_matching_result,
    )
