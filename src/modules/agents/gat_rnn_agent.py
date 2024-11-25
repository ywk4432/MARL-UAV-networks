import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_
from ..layer import GATLayer


class GATRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        """
        Args:
            input_shape (_type_): 输入的形状，即 obs_size。
            args (_type_): 包含模型配置参数的对象。
        """
        super().__init__()
        self.args = args

        # 图注意力融合编码器
        self.gat_fc = nn.Linear(input_shape, args.gat_embed_dim)
        self.gat = GATLayer(
            args.gat_embed_dim,
            args.gat_hidden_dim,
            args.gat_embed_dim,
            args.gat_head_num,
        )

        self.fc1 = nn.Linear(
            input_shape + args.gat_embed_dim, args.rnn_hidden_dim
        )  # 192*64
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 64*64
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)  # 64*30

        # 根据模型的配置参数进行一些初始化操作。
        # 初始化神经网络层的，包括层归一化（Layer Normalization）和权重矩阵的正交初始化。
        # 如果设置了'use_layer_norm'，创建一个 LayerNorm 的实例。LayerNorm 是一种用于对神经网络层进行归一化的技术，有助于加速训练和提高模型的鲁棒性。
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        # 如果设置了'use_orthogonal'，使用正交初始化方法初始化权重矩阵。正交初始化是一种初始化权重的方法，旨在避免梯度消失或梯度爆炸问题，并有助于更好地训练深度神经网络。
        # orthogonal_init_ 函数通常用于初始化权重矩阵，而 gain 参数用于调整初始化的增益。
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, mask):
        """
        Args:
            inputs (_type_): 输入的观测信息，形状为 (batch_size, n_agents, obs_dim)。
            hidden_state (_type_): 先前的隐藏状态（RNN）。
            mask: 拓扑图的邻接矩阵

        Returns:
            _type_: _description_
        """
        att, _ = self.gat(self.gat_fc(inputs), mask)
        inputs = torch.cat([inputs, att], dim=-1)
        b, a, e = inputs.size()

        # 将输入重塑为 (batch_size * n_agents, obs_dim) 的形状，以便送入全连接层。
        inputs = inputs.view(-1, e)
        # 将输入通过全连接层，并应用 ReLU 激活函数。
        # inplace=True 是指是否将操作应用到原始的张量上，而不是创建一个新的张量来存储结果。
        # 当 inplace=True 时，操作会直接修改原始张量的值，而不返回一个新的张量。这可以节省内存
        x = F.relu(self.fc1(inputs), inplace=True)
        # 将先前的隐藏状态重塑为 (batch_size * n_agents, rnn_hidden_dim)。
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # 将处理后的输入和重塑后的隐藏状态输入到 GRUCell 中，得到新的隐藏状态。
        hh = self.rnn(x, h_in)

        # 检测是否使用层归一化
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        # 将模型的输出 q 进行形状变换，其中 q 与隐藏状态的形状均变为 (batch_size, n_agents, -1)。
        return q.view(b, a, -1), hh.view(b, a, -1)
