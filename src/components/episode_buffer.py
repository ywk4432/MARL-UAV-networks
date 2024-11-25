import random
from types import SimpleNamespace as SN

import numpy as np
import torch as th

from .segment_tree import SumSegmentTree, MinSegmentTree


class EpisodeBatch:
    def __init__(
        self,
        scheme,
        groups,
        batch_size,
        max_seq_length,
        data=None,
        preprocess=None,
        device="cpu",
    ):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(
                self.scheme, self.groups, batch_size, max_seq_length, self.preprocess
            )

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        """根据 scheme 中的规范创建相应的张量，并根据预处理规范（如果有的话）更新 scheme，以便存储预处理
        后的数据。这样，EpisodeBatch 实例就能够容纳不同类型的数据，并根据定义的规范进行初始化。

        Args:
            scheme (_type_): _description_
            groups (_type_): _description_
            batch_size (_type_): _description_
            max_seq_length (_type_): _description_
            preprocess (_type_): _description_
        """
        if preprocess is not None:
            # 如果定义了预处理步骤 preprocess，则对每个预处理步骤进行处理
            for k in preprocess:
                # 对于每个预处理步骤，获取原始键 k（就是preprocess的键），新键 new_k 以及一系列转换函数 transforms
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                # 对于每个转换函数，根据当前的 vshape 和 dtype，通过 transform.infer_output_info 推断输出的新的 vshape 和 dtype
                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                # 使用新的 vshape 和 dtype 更新 self.scheme，添加新的数据字段
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {"vshape": vshape, "dtype": dtype}
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k][
                        "episode_const"
                    ]

        # 在 scheme 中添加一个名为 "filled" 的键，用于表示填充信息。该键的 vshape 为 (1,)，dtype 为 th.long
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update(
            {
                "filled": {"vshape": (1,), "dtype": th.long},
            }
        )

        # 遍历 scheme 中的字段
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(
                field_key
            )
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert (
                    group in groups
                ), "Group {} must have its number of members defined in _groups_".format(
                    group
                )
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros(
                    (batch_size, *shape), dtype=dtype, device=self.device
                )
            else:
                self.data.transition_data[field_key] = th.zeros(
                    (batch_size, max_seq_length, *shape),
                    dtype=dtype,
                    device=self.device,
                )

    def extend(self, scheme, groups=None):
        """用于扩展数据结构，可添加新的数据字段，适用于在训练过程中需要动态扩展数据结构的情况

        Args:
            scheme (_type_): _description_
            groups (_type_, optional): _description_. Defaults to None.
        """
        self._setup_data(
            scheme,
            self.groups if groups is None else groups,
            self.batch_size,
            self.max_seq_length,
        )

    def to(self, device):
        """将数据转移到指定的设备上

        Args:
            device (_type_): _description_
        """
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        """用给定的数据更新当前数据结构。支持更新 transition data 和 episode data。可以选择是否标记填充信息。

        Args:
            data (_type_): _description_
            bs (_type_, optional): 用于选择 batch 的切片. Defaults to slice(None).
            ts (_type_, optional): 用于选择时间步的切片. Defaults to slice(None).
            mark_filled (bool, optional): 一个布尔值，表示是否标记填充信息. Defaults to True.

        Raises:
            KeyError: _description_
        """
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    # 检查是否需要标记填充信息。如果需要标记填充信息，将 target["filled"][slices] 的相应位置设为 1
                    target["filled"][slices] = 1
                    # 将 mark_filled 设为 False，表示之后不再需要标记
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)  # 8*282
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        """确保张量的形状可以安全地通过 view 操作调整为目标形状。

        Args:
            v (_type_): _description_
            dest (_type_): _description_

        Raises:
            ValueError: _description_
        """
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError(
                        "Unsafe reshape of {} to {}".format(v.shape, dest.shape)
                    )
            else:
                idx -= 1

    def __getitem__(self, item):
        """支持按键名或者切片的方式获取数据。

        - 如果传入的是字符串，返回对应的数据张量。
        - 如果传入的是元组，返回新的 EpisodeBatch 对象，仅包含指定的数据字段。
        - 如果传入的是切片，返回新的 EpisodeBatch 对象，仅包含切片后的数据。

        Args:
            item (_type_): _description_

        Raises:
            ValueError: _description_
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {
                self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                for key in item
                if "group" in self.scheme[key]
            }
            ret = EpisodeBatch(
                new_scheme,
                new_groups,
                self.batch_size,
                self.max_seq_length,
                data=new_data,
                device=self.device,
            )
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(
                self.scheme,
                self.groups,
                ret_bs,
                ret_max_t,
                data=new_data,
                device=self.device,
            )
            return ret

    def _get_num_items(self, indexing_item, max_size):
        """获取切片中的元素数量

        Args:
            indexing_item (_type_): _description_
            max_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _new_data_sn(self):
        """返回一个新的 SN 对象

        Returns:
            SN: 新的 SN 对象
        """
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        """解析切片，将传入的切片规范化

        Args:
            items (_type_): _description_

        Raises:
            IndexError: _description_

        Returns:
            _type_: _description_
        """
        parsed = []
        # Only batch slice given, add full time slice
        if (
            isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            # [a,b,c]
            or (
                isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor))
            )
        ):
            # 将其封装为包含两个元素的元组，第一个元素为 items，第二个元素为 slice(None)，表示对全部时间步的切片。
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            # 对于可迭代对象，确保时间索引是连续的，如果不是连续的则引发 IndexError。
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            # TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # 如果是整数，将其转换为包含该整数范围的切片对象（slice(item, item+1)）。
                # Convert single indices to slices
                parsed.append(slice(item, item + 1))
            else:
                # 对于其他情况，保持不变。
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        """找到整个经验批次中填充时间步的最大数量

        Returns:
            _type_: _description_
        """
        # self.data.transition_data["filled"]：这是经验批次中记录每个时间步是否填充的信息。形状应该是 (batch_size, max_episode_length)，其中每个元素是一个二进制值，表示对应时间步是否填充。
        # th.sum(self.data.transition_data["filled"], 1)：在每个批次中对填充的时间步求和，得到一个大小为 (batch_size,) 的张量。每个元素表示对应批次的填充时间步总数。
        # .max(0)[0]：取所有批次的填充时间步总数中的最大值。这将返回一个包含一个元素的张量，表示所有批次中填充时间步的最大数量。
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        """返回描述性字符串，包括批次大小、最大序列长度、数据键和数据分组等信息

        Returns:
            _type_: _description_
        """
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(
            self.batch_size, self.max_seq_length, self.scheme.keys(), self.groups.keys()
        )


class ReplayBuffer(EpisodeBatch):
    def __init__(
        self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"
    ):
        super(ReplayBuffer, self).__init__(
            scheme,
            groups,
            buffer_size,
            max_seq_length,
            preprocess=preprocess,
            device=device,
        )
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(
                ep_batch.data.transition_data,
                slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                slice(0, ep_batch.max_seq_length),
                mark_filled=False,
            )
            self.update(
                ep_batch.data.episode_data,
                slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
            )
            self.buffer_index = self.buffer_index + ep_batch.batch_size
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(
                self.episodes_in_buffer, batch_size, replace=False
            )
            return self[ep_ids]

    def uni_sample(self, batch_size):
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            # Uniform sampling
            return self.uni_sample(batch_size)
        else:
            # Return the latest
            return self[self.buffer_index - batch_size : self.buffer_index]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(
            self.episodes_in_buffer,
            self.buffer_size,
            self.scheme.keys(),
            self.groups.keys(),
        )


# Adapted from the OpenAI Baseline implementations (https://github.com/openai/baselines)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        scheme,
        groups,
        buffer_size,
        max_seq_length,
        alpha,
        beta,
        t_max,
        preprocess=None,
        device="cpu",
    ):
        super(PrioritizedReplayBuffer, self).__init__(
            scheme,
            groups,
            buffer_size,
            max_seq_length,
            preprocess=preprocess,
            device="cpu",
        )
        self.alpha = alpha
        self.beta_original = beta
        self.beta = beta
        self.beta_increment = (1.0 - beta) / t_max
        self.max_priority = 1.0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

    def insert_episode_batch(self, ep_batch):
        # TODO: convert batch/episode to idx?
        pre_idx = self.buffer_index
        super().insert_episode_batch(ep_batch)
        idx = self.buffer_index
        if idx >= pre_idx:
            for i in range(idx - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority**self.alpha
                self._it_min[pre_idx + i] = self.max_priority**self.alpha
        else:
            for i in range(self.buffer_size - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority**self.alpha
                self._it_min[pre_idx + i] = self.max_priority**self.alpha
            for i in range(self.buffer_index):
                self._it_sum[i] = self.max_priority**self.alpha
                self._it_min[i] = self.max_priority**self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.episodes_in_buffer - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, t):
        assert self.can_sample(batch_size)
        self.beta = self.beta_original + (t * self.beta_increment)

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.episodes_in_buffer) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.episodes_in_buffer) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return self[idxes], idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.episodes_in_buffer
            self._it_sum[idx] = priority**self.alpha
            self._it_min[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)
