from .policy import register
from .baselines import *

register("initial-view", InitialView)
register("front-view", FrontView)
register("top-view", TopView)
