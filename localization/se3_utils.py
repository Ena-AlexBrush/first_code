import numpy as np

def se3_from_xyz_rpy(x, y, z, roll, pitch, yaw):
    """
    Build a 4x4 SE(3) homogeneous transform from translation (x, y, z)
    and roll/pitch/yaw angles (in radians), using ZYX (yaw-pitch-roll) order.

    The transform maps points from the local frame into the parent frame:
    p_parent = T_parent_from_local @ [p_local; 1].

    Input: point in local frame
    Output: point in parent frame
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
    T[:3, :3] = R            # top-left: rotation
    T[:3, 3] = [x, y, z]     # top-right: translation
    return T


def transform_points_se3(T, pts_3d):
    """
    Apply a 4x4 SE(3) transform T to an array of 3D points pts_3d (N x 3).
    Returns the transformed 3D points (N x 3).

    p_out = T @ [p_in; 1]
    """
    pts_3d = np.asarray(pts_3d) # Ensure input is a NumPy array
    N = pts_3d.shape[0] # Number of points

    # Convert to homogeneous coordinates: [x, y, z, 1]
    homog = np.hstack([pts_3d, np.ones((N, 1))])

    # Apply transform to all points at once
    out = (T @ homog.T).T # Every point is now rotated and translated into the parent frame.

    # Drop the homogeneous coordinate, keep x, y, z 
    return out[:, :3] # Drop the homogeneous coordinate, keep x, y, z


# se3 -> 2d array of pixels that give you points -> combine that with yolac ->
# so are we getting new arrays every second? Because the arrays are coming from the picture?

# Idea for combining the 2d array of pixels with depth and color -> make a new 3D/2D array with color and depth

"""
Extra comments:

Parent vs local frames:

Even though the camera is physically attached to the vehicle, it has its own 
coordinate frame and origin, which usually does NOT coincide with the vehicle's 
center. Additionally, the camera axes are oriented differently from the vehicle axes.

- Vehicle frame: origin at car center, X = forward, Y = left, Z = up
- Camera frame: origin at lens, X = right, Y = down, Z = forward (OpenCV convention)

Consequences:
- A cone 5m in front of the camera is NOT 5m in front of the car center if the 
  camera is mounted ahead of the vehicle origin.
- Axes are rotated differently, so the "forward" in camera frame isn't "forward" 
  in vehicle frame.

Solution:
1. Remap camera axes to align with vehicle axes.
2. Apply SE(3) transform using camera mounting position & orientation.

This gives accurate cone positions **relative to the car center**, which is 
essential for mapping and planning.

Donkey version:
- Imagine holding a flashlight 1m in front of your body.
- A bug is 3m ahead of the flashlight.
- Relative to your body, the bug is only 2m ahead.
- SE(3) transform does exactly this for every point the camera sees.
"""
