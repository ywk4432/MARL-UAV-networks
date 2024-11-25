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


# 测试示例
lst = [0, 0, 5, 0, 10, 0, 0, 3]
M = 5
print(top_m_indices(lst, M))  # 输出不重复的前 M 个最大值索引
