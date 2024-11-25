REGISTRY = {}

from .basic_central_controller import CentralBasicMAC
from .basic_controller import BasicMAC
from .conv_controller import ConvMAC
from .dop_controller import DOPMAC
from .gat_mac import GATMAC
from .lica_controller import LICAMAC
from .n_controller import NMAC
from .pointer_controller import POINTERMAC
from .ppo_controller import PPOMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY["gat_mac"] = GATMAC
REGISTRY["pointer_mac"] = POINTERMAC
