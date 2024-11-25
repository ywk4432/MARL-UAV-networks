REGISTRY = {}

from .atten_rnn_agent import ATTRNNAgent
from .central_rnn_agent import CentralRNNAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .gat_rnn_agent import GATRNNAgent
from .mlp_agent import MLPAgent
from .n_rnn_agent import NRNNAgent
from .noisy_agents import NoisyRNNAgent
from .pointer_agent import PointerAgent
from .rnn_agent import RNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .transfer_agent import TransferMLP1Agent
from .transferrable_agent import TransferAgent
from .transferrable_gru_agent import TransferGRUAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["transfer_gru"] = TransferGRUAgent
REGISTRY["transfer"] = TransferAgent
REGISTRY["transfer_mlp1"] = TransferMLP1Agent
REGISTRY["gat_rnn"] = GATRNNAgent
REGISTRY["pointer"] = PointerAgent
