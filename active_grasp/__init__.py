from .policy import register
from .baselines import *

register("initial-view", InitialView)
register("top-view", TopView)
register("top-trajectory", TopTrajectory)
register("circular-trajectory", CircularTrajectory)
