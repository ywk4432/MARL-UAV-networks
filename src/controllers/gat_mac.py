from .n_controller import NMAC


# This multi-agent controller shares parameters between agents
class GATMAC(NMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)
        adjacency = ep_batch["adjacency"][:, t, 0]
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, adjacency
        )

        return agent_outs
