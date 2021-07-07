from .policy import register
from .baselines import *

register("single-view", SingleViewBaseline)
register("top", TopBaseline)
