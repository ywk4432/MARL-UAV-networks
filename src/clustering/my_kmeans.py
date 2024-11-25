import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Point:
    def __init__(self, p_id, pos, aoi, data):
        self.id = p_id
        self.pos = pos
        self.aoi = aoi
        self.data = data
        self.ai = self.data * math.exp(self.aoi)

        self.cluster_id = -1


class Cluster:
    def __init__(self, max_n=10, cluster_id=-1, pList=None, alpha=1):
        self.max_n = max_n
        self.id = cluster_id
        self.n = 0.0
        self.points = []
        self.aoi = 0.0
        self.data = 0.0
        self.ai = 0.0
        self.pList = pList
        self.center = (-1.0, -1.0)
        self.main_p_id = -1
        self.dis_center = 0.0
        self.alpha = alpha

        self.min_ai = 0.0
        self.min_ai_idx = 0.0

    def get_num(self):
        self.n = len(self.points)
        return self.n

    def get_aoi(self):
        aoi = 0.0
        for p in self.points:
            aoi += self.pList[p].aoi

        # self.aoi = aoi / len(self.points)
        self.aoi = aoi
        return self.aoi

    def get_data(self):
        data = 0.0
        for p in self.points:
            data += self.pList[p].data

        # self.data = data / len(self.points)
        self.data = data
        return self.data

    def get_ai(self):
        ai = 0.0
        for p in self.points:
            ai += self.pList[p].ai

        # self.ai = ai / len(self.points)
        self.ai = ai
        return self.ai

    def get_min_ai(self):
        ai = [self.pList[i].ai for i in self.points]

        self.min_ai = min(ai)
        self.min_ai_idx = ai.index(min(ai))

        return self.min_ai

    def get_info(self):
        self.get_num()
        self.get_min_ai()

    def remove_max(self):
        """去掉ai最大的点"""
        max_ai, max_ai_id = 0.0, 0
        for i in self.points:
            if i == self.main_p_id:
                continue
            p = self.pList[i]
            if max_ai < p.dis_center + self.alpha * p.ai:
                max_ai_id = p.id
                max_ai = p.ai
        self.points.remove(max_ai_id)
        return max_ai_id

    def update(self, p_id):
        """更新点"""
        self.points.append(p_id)
        self.get_info()

        assert self.n <= self.max_n


def top_m_indices(lst, M):
    # 获取不为零的元素及其索引
    non_zero_indices = [(i, val) for i, val in enumerate(lst) if val > 0]

    # 如果不为零的元素少于 M 个，只获取不为零的元素数量
    num_to_select = min(M, len(non_zero_indices))

    # 按值降序排序不为零的元素，选择前 num_to_select 个的索引
    top_indices = [
        i
        for i, val in sorted(non_zero_indices, key=lambda x: x[1], reverse=True)[
            :num_to_select
        ]
    ]

    # 如果仍然不足 M 个索引，补充其他索引（可以从剩余的零元素中选）
    if len(top_indices) < M:
        zero_indices = [i for i, val in enumerate(lst) if val == 0]
        top_indices.extend(zero_indices[: M - len(top_indices)])  # 补齐缺少的索引

    return top_indices


# Balanced k-means algorithm
class BalancedKMeans:
    def __init__(self, cluster_num, max_iter, p_list):
        self.cluster_num = cluster_num
        self.max_iter = max_iter
        self.p_list = p_list
        self.p_to_cluster = []
        self.p_num = len(self.p_list)
        self.centroids = []
        self.cluster_p_max_num = self.p_num / self.cluster_num

        self.clusters = [
            Cluster(self.cluster_p_max_num, i, p_list) for i in range(self.cluster_num)
        ]

    def set_centroids(self):
        """找出前K个势力值最大的点作为中心"""
        ai_list = [p.ai for p in self.p_list]

        # # 获取前K个最大值
        # top_K_values = sorted(ai_list, reverse=True)[: self.cluster_num]

        # 找到前K个最大值的索引
        # top_K_value_index = [ai_list.index(value) for value in top_K_values]
        top_K_value_index = top_m_indices(ai_list, self.cluster_num)

        # 获取前K个最大值的坐标
        self.centroids = [
            (self.p_list[i].id, self.p_list[i].pos[0], self.p_list[i].pos[1])
            for i in top_K_value_index
        ]

        for i, cluster in enumerate(self.clusters):
            p = self.p_list[top_K_value_index[i]]
            cluster.points.append(p.id)
            cluster.main_p_id = p.id
            cluster.min_ai = p.ai
            cluster.n += 1
            cluster.center = (p.pos[0], p.pos[1])
            p.cluster_id = i

        if len(self.p_to_cluster) == 0:
            print("p_to_cluster is clear.")
        for centroid in self.centroids:
            if centroid[0] not in self.p_to_cluster:
                print("centroid[0]", centroid[0])
                print("p_to_cluster", self.p_to_cluster)
            self.p_to_cluster.remove(centroid[0])

    def centroids_update(self):
        """更新中心点"""
        for cluster in self.clusters:
            ai_sum = 0.0
            x_ai_sum = 0.0
            y_ai_sum = 0.0
            for p_id in cluster.points:
                p = self.p_list[p_id]
                ai_sum += p.ai
                x_ai_sum += p.ai * p.pos[0]
                y_ai_sum += p.ai * p.pos[1]

            if math.isclose(ai_sum, 0):
                x_sum = 0.0
                y_sum = 0.0
                for p_id in cluster.points:
                    p = self.p_list[p_id]
                    x_sum += p.pos[0]
                    y_sum += p.pos[1]
                n = cluster.get_num()
                cluster.center = (x_sum / n, y_sum / n)
            else:
                cluster.center = (x_ai_sum / ai_sum, y_ai_sum / ai_sum)

            for i in self.centroids:
                if self.p_list[i[0]].cluster_id == cluster.id:
                    i = (i[0], cluster.center[0], cluster.center[1])
                    break

    def set_p_to_cluster(self):
        """确定待分类的points"""
        for p in self.p_list:
            self.p_to_cluster.append(p.id)

    def fit(self):
        self.set_p_to_cluster()
        self.set_centroids()

        for _ in range(self.max_iter):
            while self.p_to_cluster != []:
                p = self.p_list[self.p_to_cluster[0]]

                # 获取与各cluster中心点的距离
                dis_to_centers = [
                    math.sqrt((p.pos[0] - center[1]) ** 2 + (p.pos[1] - center[2]) ** 2)
                    for center in self.centroids
                ]
                # 根据距离排序
                sorted_indices = [
                    index
                    for index, value in sorted(
                        enumerate(dis_to_centers), key=lambda x: x[1]
                    )
                ]

                # 确定cluster
                for center_id in sorted_indices:
                    cluster_id = self.p_list[self.centroids[center_id][0]].cluster_id
                    cluster = self.clusters[cluster_id]

                    if cluster.n < cluster.max_n:  # 该类未满
                        cluster.update(p.id)
                        p.dis_center = dis_to_centers[cluster_id]
                        self.p_to_cluster.remove(p.id)
                        break
                    elif p.ai < cluster.get_min_ai():  # 该类中存在比当前点差的点
                        # 先剔除该cluster中ai大的点，再加入新点
                        remove_p_id = cluster.remove_max()
                        self.p_to_cluster.append(remove_p_id)
                        cluster.update(p.id)
                        p.dis_center = dis_to_centers[cluster_id]
                        self.p_to_cluster.remove(p.id)
                        break

            self.centroids_update()

    def show(self):
        aoi_list = []
        data_list = []
        print("id \t ai \t aoi \t data \t pos")
        for cluster in self.clusters:
            aoi_list.append(cluster.get_aoi())
            data_list.append(cluster.get_data())
            print(
                cluster.id,
                "\t",
                cluster.get_ai(),
                "\t",
                cluster.get_aoi(),
                "\t",
                cluster.get_data(),
                "\t",
                cluster.center,
            )
        return aoi_list, data_list

    def fig_plot():
        colors = ["b", "c", "g", "k", "m", "r", "w"]

        # Visualization
        plt.figure(figsize=(12, 6))

        # Before clustering
        plt.subplot(1, 2, 1)
        plt.scatter(locations[:, 0], locations[:, 1], c="gray")
        plt.title("Before Clustering")
        plt.xlabel("X")
        plt.ylabel("Y")

        # After clustering
        plt.subplot(1, 2, 2)

        for cluster in bkm.clusters:
            plt.scatter(
                cluster.center[0],
                cluster.center[1],
                label=f"Cluster {cluster.id}",
                marker="x",
                s=50,
                color=colors[cluster.id],
            )
            cluster_points = []
            for p in cluster.points:
                cluster_points.append(list(bkm.p_list[p].pos))
            cluster_points = np.array(cluster_points)
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster.id]
            )
        plt.title("After Clustering")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        # 创建fig文件夹
        output_dir = "src/clustering/fig"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, "my_kmeans.png")
        plt.tight_layout()
        plt.savefig(output_file)


def get_other_cluster_result(bkm, name, k, labels):
    clusters = [[] for _ in range(k)]
    for i, cluster_id in enumerate(labels):
        clusters[cluster_id].append(i)
    ai_list, aoi_list, data_list = [], [], []
    for cluster in clusters:
        ai, aoi, data = 0.0, 0.0, 0.0
        for p_id in cluster:
            ai += bkm.p_list[p_id].ai
            aoi += bkm.p_list[p_id].aoi
            data += bkm.p_list[p_id].data
        ai_list.append(ai)
        aoi_list.append(aoi)
        data_list.append(data)

    # print(name)
    # print("id \t ai \t aoi \t data")
    # for i in range(k):
    #     print(
    #         i,
    #         "\t",
    #         ai_list[i],
    #         "\t",
    #         aoi_list[i],
    #         "\t",
    #         data_list[i],
    #     )

    return aoi_list, data_list, clusters


def kmeans_cluster(cluster_num, locations,bkm):
    # 应用k-means算法
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(locations)
    kmeans_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    kmeans_aoi_list, kmeans_data_list, kmeans_clusters = get_other_cluster_result(
        bkm, "k-means", cluster_num, kmeans_labels
    )

    return centroids, kmeans_data_list, kmeans_clusters


def gmm_cluster(cluster_num, locations,bkm):
    # 应用GMM算法
    gmm = GaussianMixture(n_components=cluster_num)
    gmm.fit(locations)
    gmm_labels = gmm.predict(locations)
    centroids = gmm.means_

    gmm_aoi_list, gmm_data_list ,gmm_clusters = get_other_cluster_result(
        bkm, "gmm", cluster_num, gmm_labels
    )

    return centroids, gmm_data_list,gmm_clusters


def plot_clustering_results_boxplot(bcc_results, kmeans_results, gmm_results, name):
    """
    绘制 BCC、k-means 和 GMM 三个聚类算法结果的箱线图，并使用不同颜色和样式。

    参数:
        bcc_results (list): BCC 聚类结果数据列表
        kmeans_results (list): k-means 聚类结果数据列表
        gmm_results (list): GMM 聚类结果数据列表
    """
    # 将三个数据列表放入一个数组中
    data = [bcc_results, kmeans_results, gmm_results]
    labels = ["BCC", "K-means", "GMM"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 使用蓝色、橙色和绿色
    boxprops = dict(linewidth=1.5)  # 线宽

    # 绘制箱线图
    plt.figure(figsize=(4, 3))

    # 创建每个算法的箱线图，并指定颜色和样式
    for i, (dataset, label, color) in enumerate(zip(data, labels, colors), start=1):
        plt.boxplot(
            dataset,
            positions=[i],
            widths=0.6,
            patch_artist=True,  # 使用填充颜色
            boxprops=dict(facecolor=color, color=color, **boxprops),  # 箱体
            medianprops=dict(color="black", linewidth=1.5),  # 中位数
            whiskerprops=dict(color=color, linewidth=1.5),  # 须
            capprops=dict(color=color, linewidth=1.5),  # 顶端
            flierprops=dict(marker="o", color=color, markersize=5, alpha=0.6),  # 异常值
            meanprops=dict(
                marker="D", markeredgecolor=color, markerfacecolor="white", markersize=6
            ),  # 均值
        )

    # 设置字体和标签
    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=labels,
        fontsize=14,
        fontname="Times New Roman",
    )
    plt.yticks(fontsize=10, fontname="Times New Roman")
    # plt.title("Clustering Results Comparison", fontsize=18, fontname="Times New Roman")
    # plt.xlabel("Clustering Method", fontsize=16, fontname="Times New Roman")
    plt.ylabel(f"{name}", fontsize=16, fontname="Times New Roman")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 创建fig文件夹
    output_dir = "src/clustering/fig"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"cluster_{name}.pdf")
    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":

    # Constants
    NUM_POINTS = 150
    AREA_SIZE = 30
    K = 10  # Number of clusters
    MAX_ITERATIONS = 100

    # Generate data points
    np.random.seed(0)
    locations = np.random.rand(NUM_POINTS, 2) * AREA_SIZE
    data_amount = np.random.uniform(
        30, 100, NUM_POINTS
    )  # Random float between 0 and 10
    info_age = np.random.randint(3, 10, NUM_POINTS)  # Random int between 0 and 3

    # Calculate artificial potential
    k_factor = 1.0  # Arbitrary constant k
    potentials = data_amount * np.exp(info_age)

    # Fit balanced k-means
    p_list = [
        Point(i, locations[i], info_age[i], data_amount[i]) for i in range(NUM_POINTS)
    ]
    bkm = BalancedKMeans(
        cluster_num=K,
        max_iter=MAX_ITERATIONS,
        p_list=p_list,
    )
    bkm.fit()
    bkm_aoi_list, bkm_data_list = bkm.show()

    kmeans_aoi_list, kmeans_data_list, kmeans_clusters = kmeans_cluster(K, locations)

    plot_clustering_results_boxplot(
        bkm_aoi_list, kmeans_aoi_list, gmm_aoi_list, name="AoI"
    )
    plot_clustering_results_boxplot(
        bkm_data_list, kmeans_data_list, gmm_data_list, name="Data Volume"
    )
