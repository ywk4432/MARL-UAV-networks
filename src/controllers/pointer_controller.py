import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


# This multi-agent controller shares parameters between agents
class POINTERMAC:
    def __init__(self, scheme, groups, args, env_info):
        """
        初始化是否需要更改？
        """
        self.n_agents = args.n_agents
        self.args = args
        self.env_info = env_info
        input_shape = self._get_input_shape(scheme)
        self._build_agents()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        tour_idx, tour_logp = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # Optionally, you can still use the log probabilities for further processing
        return tour_idx

    def forward(self, ep_batch, t, test_mode=False):
        """
        需要对齐输入
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        # agent_outs, agent_outs1, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        tour_idx, agent_logp = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        # assert self.agent_output_type == 'pi_logits'

        # if getattr(self.args, "mask_before_softmax", True):
        #     # Make the logits for unavailable actions very negative to minimise their affect on the softmax
        #     reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
        #     agent_outs[reshaped_avail_actions == 0] = -1e5

        # agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return tour_idx, agent_logp

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self):
        # self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.agent = agent_REGISTRY[self.args.agent](
            self.env_info["l_s_obs_shape"],
            self.env_info["l_d_obs_shape"],
            self.args.l_args["hidden_size"],
            self.args.env_args["fuav_num"],
            self.args.env_args,
            self.args.device,
        )

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        # if self.args.obs_agent_id:
        #     inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
