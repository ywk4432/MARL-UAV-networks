import torch as th
import torch.nn as nn
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class TransferGRUAgent(nn.Module):
    def __init__(self, input_shape, args, abs=True):
        """在初始化方法中，定义了三个神经网络层：一个全连接层 (self.fc1)，一个 GRUCell 循环神经网络层 (self.rnn)，
        和一个输出层 (self.fc2)。根据模型的配置参数，可能还会初始化层归一化和权重矩阵的正交初始化。

        Args:
            input_shape (_type_): 输入的形状，用于确定 nn.Linear 的输入维度。
            args (_type_): 包含模型配置参数的对象。
        """
        super(TransferGRUAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.abs = abs
        self.cat_input_dim = (
            args.cat_output_dim + args.vector_size + args.n_actions + args.n_agents
        )

        self.new_input_shape = (
            input_shape - args.vector_size - args.n_actions - args.n_agents
        )
        self.fc1 = nn.Linear(self.new_input_shape, args.cat_output_dim)  # 400*40
        self.rnn = nn.GRUCell(self.cat_input_dim, self.args.rnn_hidden_dim)  # 45*32
        # self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions)  # 64*30

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.cat_input_dim, args.hypernet_embed_agent),
            nn.ReLU(inplace=True),
            nn.Linear(
                args.hypernet_embed_agent,
                self.args.rnn_hidden_dim * self.args.n_actions,
            ),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(self.cat_input_dim, self.args.n_actions)
        )

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

    def pos_func(self, x):
        return th.abs(x)

    def forward(self, inputs, hidden_state):
        """_summary_

        Args:
            inputs (_type_): 输入的观测信息，形状为 (batch_size, n_agents, obs_dim)。
            hidden_state (_type_): 先前的隐藏状态（RNN）。

        Returns:
            _type_: _description_
        """
        b, a, e = inputs.size()

        # 将输入重塑为 (batch_size * n_agents, obs_dim) 的形状，以便送入全连接层。
        inputs = inputs.view(-1, e)

        new_inputs = inputs[:, : self.new_input_shape]
        # x = F.relu(self.fc1(new_inputs), inplace=True)
        x = self.fc1(new_inputs)
        # 将全连接层的输出和剩余的 5 个元素拼接起来
        concatenated_inputs = th.cat([x, inputs[:, self.new_input_shape :]], dim=1)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(concatenated_inputs, h_in)

        w1 = self.hyper_w1(concatenated_inputs).view(
            -1, self.args.rnn_hidden_dim, self.args.n_actions
        )  # b * t, n_agents, emb
        b1 = self.hyper_b1(concatenated_inputs).view(-1, 1, self.args.n_actions)

        if self.abs:
            w1 = self.pos_func(w1)

        # 检测是否使用层归一化
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = th.matmul(hh.unsqueeze(1), w1) + b1
        q = q.squeeze(1)

        # 将模型的输出 q 进行形状变换，其中 q 与隐藏状态的形状均变为 (batch_size, n_agents, -1)。
        return q.view(b, a, -1), hh.view(b, a, -1)
