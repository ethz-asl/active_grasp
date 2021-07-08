from .policy import register
from .baselines import *

register("single-view", SingleViewBaseline)
register("top", TopBaseline)
register("random", RandomBaseline)
register("fixed-trajectory", FixedTrajectoryBaseline)
