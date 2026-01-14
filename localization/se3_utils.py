# localization/se3_utils.py
import numpy as np

def se3_from_xyz_rpy(x, y, z, roll, pitch, yaw):
    """
    Build a 4x4 SE(3) homogeneous transform from translation (x, y, z)
    and roll/pitch/yaw angles (in radians).
    This can represent, for example, the camera pose in the vehicle frame.
    """

    # precompute sines and cosines of the Euler angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotation about Z (yaw), then Y (pitch), then X (roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    # Combined rotation: R = Rz * Ry * Rx (ZYX convention)
    R = Rz @ Ry @ Rx

    # Build 4x4 homogeneous transform:
    # [ R  t ]
    # [ 0  1 ]
    T = np.eye(4)
    T[:3, :3] = R # top-left: rotation
    T[:3, 3] = [x, y, z] # top-right: translation
    return T

def transform_points_se3(T, pts_3d):
    """
    Apply a 4x4 SE(3) transform T to an array of 3D points pts_3d (N x 3).
    Returns the transformed 3D points (N x 3).
    """
    # Number of points
    N = pts_3d.shape[0]

    # Convert to homogeneous coordinates: [x, y, z, 1]
    homog = np.hstack([pts_3d, np.ones((N, 1))])

    # Apply transform to all points at once
    out = (T @ homog.T).T

    # Drop the homogeneous coordinate, keep x, y, z
    return out[:, :3]

# se3 -> 2d array of pixels that give you points -> combine that with yolac ->
# so are we getting new arrays every second? Because the arrays are coming from the picture?

# Idea for combining the 2d array of pixels with depth and color -> make a new 3D/2D array with color and depth