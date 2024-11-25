import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")


class PointerAgent(nn.Module):
    def __init__(
        self,
        static_size,
        dynamic_size,
        hidden_size,
        sequence_size,
        env_args,
        device="cpu",
        update_fn=None,
        mask_fn=None,
        num_layers=1,
        dropout=0.0,
    ):
        super(PointerAgent, self).__init__()
        self.pointer_network = DRL4TSP(
            static_size,
            dynamic_size,
            hidden_size,
            sequence_size,
            env_args,
            update_fn,
            mask_fn,
            num_layers,
            dropout,
            device,
        )
        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.sequence_size = sequence_size
        self.device = device

    def forward(self, inputs, last_hh=None):
        static, dynamic = self.prepare_inputs(inputs)
        # 第三个变量原来是x0 static, dynamic, x0 = batch
        tour_idx, tour_logp = self.pointer_network(
            static,
            dynamic,
        )
        return tour_idx, tour_logp

    def prepare_inputs(self, inputs):
        # 转换inputs

        static = inputs[:, : self.static_size].reshape(
            -1, int(self.static_size / self.sequence_size), self.sequence_size
        )
        dynamic = inputs[:, self.static_size :].reshape(
            -1, int(self.dynamic_size / self.sequence_size), self.sequence_size
        )
        return static, dynamic

    def init_hidden(self):
        return None


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, device):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.device = device

    def forward(self, input):

        if input.device != self.device:
            input = input.to(self.device)
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(
            torch.zeros((1, 1, hidden_size), device=device, requires_grad=True)
        )

        self.W = nn.Parameter(
            torch.zeros(
                (1, hidden_size, 3 * hidden_size), device=device, requires_grad=True
            )
        )

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(
            torch.zeros((1, 1, hidden_size), device=device, requires_grad=True)
        )

        self.W = nn.Parameter(
            torch.zeros(
                (1, hidden_size, 2 * hidden_size), device=device, requires_grad=True
            )
        )

        # Used to compute a representation of the current decoder output
        # GRU（输入dim，隐含层dim，层数） 处理解码器输出？
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)  # 移除张量中大小为 1 的维度。

        # Always apply dropout on the RNN output ，防止过拟合
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        # static_hidden 对应 ej  rnn_out 对应解码器的隐藏状态 dt
        enc_attn = self.encoder_attn(
            static_hidden, dynamic_hidden, rnn_out
        )  # 注意力机制计算得分
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(
        self,
        static_size,
        dynamic_size,
        hidden_size,
        sequence_size,
        env_args,
        update_fn=None,
        mask_fn=None,
        num_layers=1,
        dropout=0.0,
        device="cpu",
    ):
        super(DRL4TSP, self).__init__()
        if dynamic_size < 1:
            raise ValueError(
                ":param dynamic_size: must be > 0, even if the "
                "problem has no dynamic elements"
            )
        # update_fn：更新动态特征。例如，在 VRP 中，当车辆访问一个节点时，需求和负载会动态变化。
        # mask_fn：用于确定哪些节点可以在下一步被选择，避免重复访问已经访问过的节点
        # self.vrp_dataset = VehicleRoutingDataset(num_samples=0, input_size=0)
        # self.update_fn = self.vrp_dataset.update_dynamic
        # self.mask_fn = self.vrp_dataset.update_mask
        self.device = device
        # Define the encoder & decoder models 编码是一次卷积 初始化编码解码指针网络
        self.static_encoder = Encoder(
            int(static_size / sequence_size), hidden_size, self.device
        )
        self.dynamic_encoder = Encoder(
            int(dynamic_size / sequence_size), hidden_size, self.device
        )
        self.decoder = Encoder(
            int(static_size / sequence_size), hidden_size, self.device
        )
        self.pointer = Pointer(hidden_size, num_layers, dropout)
        # 初始化模型的参数
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified 解码器的初始输入状态
        self.x0 = torch.zeros(
            (1, int(static_size / sequence_size), 1), requires_grad=True, device=device
        )
        self.sequence_size = sequence_size
        self.env_args = env_args

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        # batch_size, input_size= static.size()

        batch_size, input_size, seq_size = static.size()
        assert seq_size == self.sequence_size

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1).to(self.device)

        # Always use a mask - if no function is provided, we don't update it
        # 屏蔽已经访问过的节点
        mask = torch.ones(batch_size, self.sequence_size, device=self.device)

        # Structures for holding the output sequences 访问的节点顺序和每一步概率
        tour_idx, tour_logp = [], []
        max_steps = self.sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        # 编码过程 只需要一次
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for uav_id in range(max_steps):

            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)
            # 前一步的隐藏状态（last_hh）（pointnetwork输出的）保持时序信息 和 当前输入的解码器输出（decoder_hidden）
            probs, last_hh = self.pointer(
                static_hidden, dynamic_hidden, decoder_hidden, last_hh
            )
            probs = F.softmax(probs + mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation 更新动态特征
            if self.update_fn is not None:
                if self.env_args["updata_fn"] == "new":
                    dynamic = self.new_update_fn(dynamic, ptr.data, uav_id)
                else:
                    dynamic = self.update_fn(
                        dynamic, ptr.data, copy.deepcopy(static), uav_id
                    )
                # dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()  # 判断任务是否已经完成
                logp = logp * (1.0 - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            # 提取出的城市特征作为下一步解码器的输入
            decoder_input = torch.gather(
                static, 2, ptr.view(-1, 1, 1).expand(-1, input_size, 1)
            ).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len) 节点序列
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)节点对数概率序列

        return tour_idx, tour_logp

    def new_update_fn(self, dynamic, chosen_idx, uav_id):
        """Updates the (demand, remaining formations) dataset values"""

        cov = dynamic[:, 0].clone()
        connect = dynamic[:, 1].clone()

        for i in range(chosen_idx.size(0)):
            if chosen_idx[i] >= 0:  # If a waypoint is chosen
                cov[i, uav_id] = 0
                connect[i, chosen_idx[i]] = 1

        tensor = torch.cat(
            (
                cov.unsqueeze(1),
                connect.unsqueeze(1),
            ),
            1,
        )
        return tensor

    def update_fn(self, dynamic, chosen_idx, static, uav_id):
        """Updates the (demand, remaining formations) dataset values"""
        posx = dynamic[:, 0].clone()  # (batch_size, seq_len)
        posy = dynamic[:, 1].clone()
        energy = dynamic[:, 2].clone()
        cov = dynamic[:, 3].clone()
        connect = dynamic[:, 4].clone()

        # Update demands based on chosen indices
        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        for i in range(chosen_idx.size(0)):
            if chosen_idx[i] >= 0:  # If a waypoint is chosen
                x1 = copy.deepcopy(posx[i, uav_id])
                y1 = copy.deepcopy(posy[i, uav_id])

                x2 = copy.deepcopy(static[0, 0, chosen_idx[i]])
                y2 = copy.deepcopy(static[0, 1, chosen_idx[i]])

                posx[i, uav_id] = static[0, 0, chosen_idx[i]]
                posy[i, uav_id] = static[0, 1, chosen_idx[i]]
                energy[i, uav_id] -= self.energy_update(x1, x2, y1, y2)
                cov[i, uav_id] = 1

        for i in range(chosen_idx.size(0)):
            if chosen_idx[i] >= 0:  # If a waypoint is chosen
                connect[i, chosen_idx[i]] = 1

        tensor = torch.cat(
            (
                posx.unsqueeze(1),
                posy.unsqueeze(1),
                energy.unsqueeze(1),
                cov.unsqueeze(1),
                connect.unsqueeze(1),
            ),
            1,
        )
        return tensor

    def mask_fn(self, mask, dynamic, chosen_idx):
        """Marks the visited city, so it can't be selected a second time."""
        mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
        return mask

    def energy_update(self, x_1, x_2, y_1, y_2):

        x = math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

        c_1 = self.env_args["flight_c_1"]
        c_2 = self.env_args["flight_c_2"]
        c_3 = self.env_args["flight_c_3"]
        energy_cost = (
            self.env_args["flight_cost"]
            + c_1 * x
            + c_2 * math.pow(x, 2)
            + c_3 * math.pow(x, 3)
        )

        u_energy_cost = energy_cost / self.env_args["fuav_init_energy"]
        return energy_cost


# class VehicleRoutingDataset(Dataset):
class VehicleRoutingDataset:
    def __init__(self, num_samples, input_size, max_demand=9, seed=None):
        super(VehicleRoutingDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.max_demand = max_demand

        # Locations (starting point + waypoints)
        locations = torch.rand(
            (num_samples, 2, input_size + 1)
        )  # +1 for starting point
        self.static = locations

        # Initialize demands for waypoints
        dynamic_shape = (num_samples, 1, input_size + 1)  # +1 for remaining formations
        demands = torch.randint(1, max_demand + 1, dynamic_shape).float() / float(
            max_demand
        )
        demands[:, 0, 0] = 0  # Starting point demand

        # Remaining formations initialized to a constant value (e.g., number of formations)
        self.remaining_formations = torch.full(
            (num_samples, 1, input_size + 1), 1
        )  # Can adjust as needed

        # Combine dynamic features: demands and remaining formations
        self.dynamic = torch.cat((demands, self.remaining_formations), dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return static features, dynamic features, and starting location
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_dynamic(dynamic, chosen_idx):
        """Updates the (demand, remaining formations) dataset values"""
        demands = dynamic[:, 0].clone()  # (batch_size, seq_len)
        formations = dynamic[:, 1].clone()  # (batch_size, seq_len)

        # Update demands based on chosen indices
        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        for i in range(chosen_idx.size(0)):
            if chosen_idx[i] > 0:  # If a waypoint is chosen
                demands[i, chosen_idx[i]] = 0  # Set 0 for visited waypoint
                formations[i, chosen_idx[i]] = 0
        tensor = torch.cat((demands.unsqueeze(1), formations.unsqueeze(1)), 1)
        return tensor

    def update_mask(mask, dynamic, chosen_idx):
        """Marks the visited city, so it can't be selected a second time."""
        mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
        return mask

    # def update_mask(self, mask, dynamic, chosen_idx=None):
    #     """Updates the mask used to hide non-valid states."""

    #     demands = dynamic.data[:, 0]  # (batch_size, seq_len)

    #     # If there are no remaining demands, we can end the tour.
    #     if demands.eq(0).all():
    #         return demands * 0.

    #     # Create a mask for valid demands
    #     new_mask = demands.ne(0)

    #     # Ensure we don't revisit already visited waypoints
    #     visited = chosen_idx.ne(0)
    #     new_mask[visited.nonzero()] = 0

    #     return new_mask.float()


if __name__ == "__main__":
    raise Exception("Cannot be called from main")
