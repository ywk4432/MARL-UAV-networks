from .local_learner import Learner as LocalLearner
from .pointer_learner import Learner as pointerLearner
from .q_learner import Learner as QLearner
from .two_learner import Learner as TwoLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["two_learner"] = TwoLearner
REGISTRY["local_learner"] = LocalLearner
REGISTRY["pointer_learner"] = pointerLearner
