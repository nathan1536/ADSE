from task1 import get_aabb_polygon, generate_trajectories
from matplotlib import pyplot as plt
import random
import shapely.ops
import trajectory_planning_helpers as tph
from shapely.geometry import Polygon
from shapely.ops import unary_union

width = 1.5
length = 4.0

trajectory1, trajectory2 = generate_trajectories(n=2, vehicle_width=width, vehicle_length=length)
get_aabb_polygon(trajectory=trajectory1)

plot_polygon(polygon, ax=ax, add_points=False, color=RED)
plot_points(polygon, ax=ax, color=GRAY, alpha=0.7)
