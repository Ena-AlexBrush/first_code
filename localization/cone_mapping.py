"""
NOTE:
This file is meant to be imported, not run directly.
Run pose_playground.py from the project root instead.
"""

import polars as pl
import numpy as np
from localization.se2_utils import se2_from_xytheta, transform_points

def cones_vehicle_to_global(
    pose_df: pl.DataFrame,
    cones_vehicle: np.ndarray,
    sample_every: int = 1,
) -> pl.DataFrame:
    rows = []
    sample = pose_df[::sample_every]
    for pose_id, row in enumerate(sample.iter_rows(named=True)):
        T = se2_from_xytheta(row["x"], row["y"], row["theta"])
        pts_global = transform_points(T, cones_vehicle)
        for cone_id, (xg, yg) in enumerate(pts_global):
            rows.append({
                "pose_id": pose_id,
                "cone_id": cone_id,
                "x_global": float(xg),
                "y_global": float(yg),
            })
    return pl.DataFrame(rows)

##Old code before refactor to function

# #This list will store all cone observations
# #Each entry corresponds to one cone seen at one pose
# rows = []

# #Subsample poses so we don't spam output
# K = 20
# sample = pose_df[::K]

# #row is a dictionary that has keys time_s, x, y, theta
# #pose_id = index of the pose (time step)
# for pose_id, row in enumerate(sample.iter_rows(named=True)):
#     #SE(2): vehicle frame -> world frame for this pose
#     T_world_vehicle = se2_from_xytheta(row["x"], row["y"], row["theta"])

#     #Convert relative cone positions into global frame
#     pts_global = transform_points(T_world_vehicle, pts_vehicle)

#     #Loop over each cone seen at this pose
#     for cone_id, (xg, yg) in enumerate(pts_global):
#         rows.append({
#             "pose_id": pose_id, #Which time step
#             "cone_id": cone_id, #Which cone at that time
#             "x_global": float(xg),
#             "y_global": float(yg),
#         })

# global_cones_df = pl.DataFrame(rows)
# print("First few global cone positions:")
# print(global_cones_df.head())