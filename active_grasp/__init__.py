from .policy import register
from .baselines import *
from .nbv import NextBestView

register("initial-view", InitialView)
register("top-view", TopView)
register("top-trajectory", TopTrajectory)
register("nbv", NextBestView)
