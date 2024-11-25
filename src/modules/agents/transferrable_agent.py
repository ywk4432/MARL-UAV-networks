"""
Date: 2024-01-25 19:04:16
description: xxx xxx
LastEditors: Wenke Yuan
LastEditTime: 2024-01-25 19:04:17
FilePath: /pymarl_uav/src/modules/agents/transferrable_agent.py
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class TransferAgent(nn.Module):
    def __init__(self, input_shape, args, abs=True):
        super(TransferAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents  # 1
        self.state_dim = args.env_args["state_size"]  # 5
        self.channel_num = args.env_args["channel_num"]  # 5
        self.abs = abs
        self.img_radius = args.env_args["uav"]["max_obs_radius"]  # 2

        self.img_size = 2 * self.img_radius
        self.img_input_dim = (
            self.img_size * self.img_size * args.env_args["channel_num"]
        )
        keep_dim = (
            input_shape - self.img_input_dim - self.state_dim
        )  # 原始观测中除去img和state的部分
        self.cat_input_dim = args.img_output_dim + keep_dim

        # 对原始观测输入做预处理
        self.conv1 = nn.Conv2d(
            in_channels=self.channel_num,
            out_channels=self.channel_num,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.fc1 = nn.Linear(self.img_radius * self.img_radius * 1, args.img_conv_dim)
        self.fc2 = nn.Linear(self.cat_input_dim, args.hidden_dim)

        # 由state构建 hyper w1 b1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, args.hypernet_embed_agent),
            nn.ReLU(inplace=True),
            nn.Linear(
                args.hypernet_embed_agent,
                self.args.hidden_dim * self.args.n_actions,
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
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def pos_func(self, x):
        return th.abs(x)

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        # 将输入重塑为 (batch_size * n_agents, obs_dim) 的形状，以便送入全连接层。
        inputs = inputs.view(-1, e)
        state = inputs[:, -self.state_dim :]
        img_inputs = inputs[:, : self.img_input_dim]

        # 对原始观测中的图像部分进行处理
        origin_img = img_inputs.view(
            -1, self.img_size, self.img_size, self.args.env_args["channel_num"]
        )
        origin_img = origin_img.permute(0, 3, 1, 2)
        img_conv = F.relu(self.conv1(origin_img))
        img_conv = img_conv.reshape(b * a, self.channel_num, -1)  # 5*10
        img_output = F.elu(self.fc1(img_conv))
        img_output = img_output.reshape(b * a, -1)

        # 将全连接层的输出和剩余的 5 个元素拼接起来
        cat_inputs = th.cat(
            [img_output, inputs[:, self.img_input_dim : -self.state_dim]], dim=1
        )

        x = F.elu(self.fc2(cat_inputs))  # 48*64
        X = x.unsqueeze(1)  # 48*1*64

        w1 = self.hyper_w1(state).view(
            -1, self.args.hidden_dim, self.args.n_actions
        )  # b*a, n_actions
        b1 = self.hyper_b1(state).view(-1, 1, self.args.n_actions)

        if self.abs:
            w1 = self.pos_func(w1)

        # 检测是否使用层归一化
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(x))
        else:
            q = th.bmm(X, w1) + b1

        return q.view(b, a, -1), x.view(b, a, -1)
