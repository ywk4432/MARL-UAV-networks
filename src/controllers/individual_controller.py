from .basic_individual_controller import BasicIndividualMAC


# This multi-agent controller shares parameters between agents
class NMAC(BasicIndividualMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            qvals[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)

        agent_outs = []
        for agent in self.agents:
            agent_out, self.hidden_states = agent(agent_inputs, self.hidden_states)
            agent_outs.append(agent_out)

        return agent_outs
