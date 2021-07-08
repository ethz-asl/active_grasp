from .policy import register
from .baselines import *

register("single-view", SingleView)
register("top", TopView)
register("random", RandomView)
register("fixed-trajectory", FixedTrajectory)
register("alignment", AlignmentView)
