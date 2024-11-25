from .TPRA_runner import TPRARunner
from .episode_runner import EpisodeRunner
from .gat_runner import GATRunner
from .large_timescale_runner import LargeTimsecaleRunner
from .parallel_runner import ParallelRunner
from .small_timescale_runner import SmallTimsecaleRunner
from .transfer_runner import TransferRunner
from .two_timescale_runner import TwoTimsecaleRunner
from .multi_object_runner import MultiObjectRunner
from .TP_runner import TPRunner
from .RA_runner import RARunner
from .TP_runner_new import TPRunner_new


REGISTRY = {}

REGISTRY["parallel"] = ParallelRunner
REGISTRY["transfer"] = TransferRunner
REGISTRY["two_timescale"] = TwoTimsecaleRunner
REGISTRY["gat"] = GATRunner
REGISTRY["small_timescale"] = SmallTimsecaleRunner
REGISTRY["large_timescale"] = LargeTimsecaleRunner
REGISTRY["episode"] = EpisodeRunner
REGISTRY["TPRA"] = TPRARunner
REGISTRY["multi_object"] = MultiObjectRunner
REGISTRY["TP"] = TPRunner
REGISTRY["RA"] = RARunner
REGISTRY["TP_new"] = TPRunner_new
