from .policy import register
from .baselines import *
from .nbv import *

register("single-view", SingleView)
register("top", TopView)
register("random", RandomView)
register("fixed-trajectory", FixedTrajectory)
register("alignment", AlignmentView)
register("nbv", NextBestView)
