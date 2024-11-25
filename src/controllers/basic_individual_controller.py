import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class BasicIndividualMAC:
    def __init__(self, scheme, groups, args):
        """
        - 存储智能体数量 (self.n_agents) 和参数 (self.args)。
        - 通过调用 _get_input_shape 方法获取输入形状。
        - 通过调用 _build_agents 方法构建智能体。
        - 存储智能体输出类型 (self.agent_output_type)。
        - 创建动作选择器 (self.action_selector) 并根据参数进行相应初始化。
        - 存储是否保存概率 (self.save_probs)。
        - 初始化隐藏状态为 None。

        Args:
            scheme (_type_): _description_
            groups (_type_): _description_
            args (_type_): _description_
        """
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type  # q

        # ε-greedy行为策略
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, "save_probs", False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """获取可用动作 (avail_actions)。通过调用 forward 方法获取智能体输出。使用动作选择器 (self.action_selector) 选择动作。

        Args:
            ep_batch (_type_): 表示一个时刻的批次样本。
            t_ep (_type_): 当前时间步。
            t_env (_type_): 当前环境时间。
            bs (_type_, optional): 批次索引切片，默认为全部。
            test_mode (bool, optional): 是否处于测试模式。 Defaults to False.

        Returns:
            _type_: _description_
        """
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)

        if test_mode:
            self.agent.eval()

        agent_outs = []
        for agent in self.agents:
            agent_out, self.hidden_states = agent(agent_inputs, self.hidden_states)
            agent_outs.append(agent_out)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(
                batch_size, self.n_agents, -1
            )  # bav

    def parameters(self):
        parameters = []
        for agent in self.agents:
            parameters += agent.parameters()
        return parameters

    def load_state(self, other_mac):
        for agent in self.agents:
            agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()

    def save_models(self, path):
        for agent in self.agents:
            th.save(agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        for agent in self.agents:
            agent.load_state_dict(
                th.load(
                    "{}/agent.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )

    def _build_agents(self, input_shape):
        for _ in range(self.n_agents):
            self.agents.append(agent_REGISTRY[self.args.agent](input_shape, self.args))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        # 创建了一个空列表 inputs，用于存储构建智能体输入的各个部分
        inputs = []
        # 将当前时间步 t 的观测信息添加到 inputs 中。
        # 假设观测信息的形状为 (batch_size, 1, n_agents, obs_dim)，则这一步将其变形为 (batch_size, n_agents, obs_dim)。
        inputs.append(batch["obs"][:, t])  # b1av
        # 如果设置了'obs_last_action'，添加上一个动作信息
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        # 如果设置了'obs_agent_id'，添加代理ID信息
        if self.args.obs_agent_id:
            # 使用 torch.eye 函数创建了一个单位矩阵，self.n_agents 表示矩阵的行数。device=batch.device 用于确保该矩阵位于与输入批次相同的设备上。
            # 使用 unsqueeze(0) 在第 0 维度上增加了一个维度。假设原始单位矩阵形状为 (self.n_agents, self.n_agents)，那么经过 unsqueeze(0) 后的形状为 (1, self.n_agents, self.n_agents)。
            # 使用 expand 方法在指定的维度上进行扩展。具体到这里，扩展了第 0 维度（bs 表示批次大小），而 -1 表示该维度的大小不变。因此，最终形状将变为 (bs, self.n_agents, self.n_agents)。
            # 创建一个形状为 (bs, self.n_agents, self.n_agents) 的张量，其中每个批次中的每个智能体都有一个单位矩阵。这样的矩阵常用于表示代理的标识信息，其中单位矩阵的每行对应一个智能体，表示该智能体在一个独立的标识维度上。
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )
        # 将 inputs 列表中的各个部分按照最后一个维度（特征维度）拼接在一起，得到最终的智能体输入。
        # 这样构建的输入形状为 (batch_size, n_agents, input_dim)。
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        """返回包含观测信息、上一个动作信息和代理ID信息的所有形状组成的列表。这个列表描述了智能体在一个时间步中接收的所有信息的形状，用于构建智能体的输入。

        Args:
            scheme (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape  # 156+30+6
