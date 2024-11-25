from .TPRA_run import run as TPRA_run
from .TP_run import run as TP_run
from .TPRA_per_run import run as TPRA_per_run
from .RA_per_run import run as RA_per_run
from .TP_run_new import run as TP_run_new

from .dop_run import run as dop_run
from .large_timescale_per_run import run as large_timescale_per_run
from .multi_object_run import run as multi_object_run
from .on_off_run import run as on_off_run
from .on_run import run as on_run
from .per_run import run as per_run
from .run import run as default_run
from .small_timescale_per_run import run as small_timescale_per_run
from .small_timescale_run import run as small_timescale_run
from .transfer_run import run as transfer_run
from .two_sample_run import run as two_sample_run
from .two_timescale_per_run import run as two_timescale_per_run
from .two_timescale_run import run as two_timescale_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["on_off"] = on_off_run
REGISTRY["on"] = on_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run
REGISTRY["two_sample"] = two_sample_run
REGISTRY["transfer"] = transfer_run
REGISTRY["two_timescale"] = two_timescale_run
REGISTRY["small_timescale"] = small_timescale_run
REGISTRY["small_timescale_per"] = small_timescale_per_run
REGISTRY["two_timescale_per"] = two_timescale_per_run
REGISTRY["large_timescale_per"] = large_timescale_per_run
REGISTRY["TPRA"] = TPRA_run
REGISTRY["TPRA_per"] = TPRA_per_run
REGISTRY["TP"] = TP_run
REGISTRY["TP_new"] = TP_run_new
REGISTRY["RA_per"] = RA_per_run
REGISTRY["multi_object"] = multi_object_run
