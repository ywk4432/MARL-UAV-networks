import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# 生成随机数据点
num_points = 50
area_size = 20

positions = np.random.rand(num_points, 2) * area_size
data_amount = np.random.rand(num_points) * 10
info_age = np.random.randint(0, 6, num_points)


# 计算人工势力值
def compute_influence(data_amount, info_age):
    return data_amount * np.exp(info_age)


influence_values = compute_influence(data_amount, info_age)
data_points = np.column_stack((positions, data_amount, info_age, influence_values))


# 实现均衡k-means聚类算法
def balanced_k_means(data_points, k, max_iter=10, max_points_per_cluster=10):
    centers = data_points[np.argsort(-data_points[:, 3])[:k], :2]
    cluster_assignments = np.full(len(data_points), -1, dtype=int)
    cluster_sizes = np.zeros(k, dtype=int)

    for _ in range(max_iter):
        distances = cdist(data_points[:, :2], centers)
        new_cluster_assignments = np.argmin(distances, axis=1)

        for i in range(k):
            cluster_indices = np.where(new_cluster_assignments == i)[0]
            if len(cluster_indices) > 0:
                valid_indices = cluster_indices[
                    cluster_sizes[i]
                    + np.arange(len(cluster_indices) - cluster_sizes[i])
                ]
                valid_indices = valid_indices[:max_points_per_cluster]
                cluster_assignments[valid_indices] = i
                cluster_sizes[i] = len(valid_indices)
                centers[i] = data_points[valid_indices, :2].mean(axis=0)

        if np.array_equal(new_cluster_assignments, cluster_assignments):
            break
        cluster_assignments = new_cluster_assignments

    return cluster_assignments, centers


# 标准的k-means聚类算法
def standard_k_means(data_points, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_points[:, :2])
    return kmeans.labels_, kmeans.cluster_centers_


# 设定k值
k = 5

# 计算标准k-means聚类
standard_cluster_assignments, standard_centers = standard_k_means(data_points, k)

# 计算均衡k-means聚类
balanced_cluster_assignments, balanced_centers = balanced_k_means(data_points, k)


# 计算聚类结果统计信息
def compute_cluster_stats(data_points, cluster_assignments, k):
    cluster_stats = []
    for i in range(k):
        cluster_indices = np.where(cluster_assignments == i)[0]
        num_points = len(cluster_indices)
        influence_sum = data_points[cluster_indices, 3].sum()
        data_amount_sum = data_points[cluster_indices, 2].sum()
        info_age_sum = data_points[cluster_indices, 2].sum()
        cluster_stats.append((num_points, influence_sum, data_amount_sum, info_age_sum))
    return cluster_stats


standard_stats = compute_cluster_stats(data_points, standard_cluster_assignments, k)
balanced_stats = compute_cluster_stats(data_points, balanced_cluster_assignments, k)


# 输出聚类结果统计信息
def print_cluster_stats(stats, method_name):
    df = pd.DataFrame(
        stats,
        columns=["Num Points", "Influence Sum", "Data Amount Sum", "Info Age Sum"],
    )
    df.index.name = "Cluster"
    print(f"\n{method_name} Cluster Stats:")
    print(df)


print_cluster_stats(standard_stats, "Standard K-means")
print_cluster_stats(balanced_stats, "Balanced K-means")


# 绘制聚类结果折线图
def plot_comparison(standard_stats, balanced_stats, k):
    labels = [f"Cluster {i+1}" for i in range(k)]

    standard_stats = np.array(standard_stats)
    balanced_stats = np.array(balanced_stats)

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cluster 数量
    axes[0, 0].plot(labels, standard_stats[:, 0], marker="o", label="Standard K-means")
    axes[0, 0].plot(labels, balanced_stats[:, 0], marker="o", label="Balanced K-means")
    axes[0, 0].set_title("Number of Points per Cluster")
    axes[0, 0].legend()

    # 人工势力值的和
    axes[0, 1].plot(labels, standard_stats[:, 1], marker="o", label="Standard K-means")
    axes[0, 1].plot(labels, balanced_stats[:, 1], marker="o", label="Balanced K-means")
    axes[0, 1].set_title("Sum of Influence Values per Cluster")
    axes[0, 1].legend()

    # 用户待收集数量的和
    axes[1, 0].plot(labels, standard_stats[:, 2], marker="o", label="Standard K-means")
    axes[1, 0].plot(labels, balanced_stats[:, 2], marker="o", label="Balanced K-means")
    axes[1, 0].set_title("Sum of Data Amount per Cluster")
    axes[1, 0].legend()

    # 信息年龄的和
    axes[1, 1].plot(labels, standard_stats[:, 3], marker="o", label="Standard K-means")
    axes[1, 1].plot(labels, balanced_stats[:, 3], marker="o", label="Balanced K-means")
    axes[1, 1].set_title("Sum of Info Age per Cluster")
    axes[1, 1].legend()

    # 创建fig文件夹
    output_dir = "src/clustering/fig"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(output_file)


plot_comparison(standard_stats, balanced_stats, k)


# 绘制聚类结果
def plot_clusters(
    data_points,
    standard_assignments,
    balanced_assignments,
    standard_centers,
    balanced_centers,
    k,
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 标准k-means聚类结果
    axes[0].scatter(
        data_points[:, 0],
        data_points[:, 1],
        c=standard_assignments,
        cmap="rainbow",
        label="Standard K-means",
    )
    axes[0].scatter(
        standard_centers[:, 0],
        standard_centers[:, 1],
        color="black",
        marker="x",
        s=100,
        label="Centers",
    )
    axes[0].set_title("Standard K-means Clustering")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].legend()

    # 均衡k-means聚类结果
    axes[1].scatter(
        data_points[:, 0],
        data_points[:, 1],
        c=balanced_assignments,
        cmap="rainbow",
        label="Balanced K-means",
    )
    axes[1].scatter(
        balanced_centers[:, 0],
        balanced_centers[:, 1],
        color="black",
        marker="x",
        s=100,
        label="Centers",
    )
    axes[1].set_title("Balanced K-means Clustering")
    axes[1].set_xlabel("X Position")
    axes[1].set_ylabel("Y Position")
    axes[1].legend()

    # 创建fig文件夹
    output_dir = "src/clustering/fig"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "clusters_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file)


plot_clusters(
    data_points,
    standard_cluster_assignments,
    balanced_cluster_assignments,
    standard_centers,
    balanced_centers,
    k,
)
