"""
Date: 2024-01-25 19:04:16
description: xxx xxx
LastEditors: Wenke Yuan
LastEditTime: 2024-01-25 19:04:17
FilePath: /pymarl_uav/src/modules/agents/transferrable_agent.py
"""

import torch as th
import torch.nn as nn
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class TransferMLP1Agent(nn.Module):
    def __init__(self, input_shape, args, abs=True):
        super(TransferMLP1Agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents  # 1
        self.state_dim = args.state_shape
        self.obs_dim = args.obs_shape
        self.x_dim = args.obs_shape
        if args.obs_agent_id:
            self.x_dim += args.n_agents
        if args.obs_last_action:
            self.x_dim += args.n_actions
        self.abs = abs

        # 由state构建 hyper w1 b1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed_agent),
            nn.ReLU(inplace=True),
            nn.Linear(
                args.hypernet_embed_agent,
                self.x_dim * self.args.n_actions,
            ),
            # nn.Sigmoid(),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(self.state_dim, self.args.n_actions),
            # nn.Sigmoid(),
        )

        # 初始化神经网络层的，包括层归一化（Layer Normalization）和权重矩阵的正交初始化。
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        # 如果设置了'use_orthogonal'，使用正交初始化方法初始化权重矩阵。
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        return None

    def pos_func(self, x):
        return th.abs(x)

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()

        # 将输入重塑为 (batch_size * n_agents, obs_dim) 的形状，以便送入全连接层。
        inputs = inputs.view(-1, e)
        x = inputs[:, : -self.state_dim]
        state = inputs[:, -self.state_dim :]

        w1 = self.hyper_w1(state).view(
            -1, self.x_dim, self.args.n_actions
        )  # b*a, n_actions
        b1 = self.hyper_b1(state).view(-1, 1, self.args.n_actions)

        if self.abs:
            w1 = self.pos_func(w1)

        X = x.unsqueeze(1)

        # 检测是否使用层归一化
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(X))
        else:
            q = th.bmm(X, w1) + b1

        return q.view(b, a, -1), None
