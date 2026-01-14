"""
Camera to Vehicle Frame Transformation

This module handles converting 3D cone positions from camera frame
to vehicle frame, then projecting to 2D for mapping.

Coordinate Frames:
- Camera (OpenCV): X = right, Y = down, Z = forward
- Vehicle:         X = forward, Y = left, Z = up
- Map:             X = forward, Y = left (2D ground plane)

Pipeline:
1. Backproject pixels + depth into OpenCV camera frame.
2. Remap OpenCV camera axes to a "camera-in-vehicle-axes" frame.
3. Apply SE(3) transform from camera to vehicle.
4. Optionally project to 2D ground plane.
"""

import numpy as np
from localization.se3_utils import se3_from_xyz_rpy, transform_points_se3


class CameraToVehicleTransform:
    """
    Manages the complete transformation pipeline from camera frame to vehicle frame.

    This class handles:
    1. Camera intrinsics (pixel to 3D ray conversion)
    2. Coordinate frame remapping (OpenCV camera axes to vehicle axes)
    3. SE(3) transformation (camera position/orientation on vehicle)
    4. Projection to 2D ground plane

    Camera Mounting Configuration (in vehicle frame):
    - Position: (x_camera, y_camera, z_camera) relative to vehicle origin
    - Orientation: (roll, pitch, yaw) angles
      * pitch_camera > 0 means camera is tilted DOWN to see the track (user convention)
      * internally converted to right-hand-rule pitch for SE(3)
    """

    def __init__(
        self,
        x_camera: float = 0.5,
        y_camera: float = 0.0,
        z_camera: float = 0.3,
        roll_camera: float = 0.0,
        pitch_camera: float = 0.0,
        yaw_camera: float = 0.0,
        camera_intrinsics: dict = None,
    ):
        """
        Initialize the camera-to-vehicle transformation.

        Args:
            x_camera: Forward distance from vehicle origin (meters)
            y_camera: Left distance from vehicle centerline (meters)
            z_camera: Height above ground (meters)
            roll_camera: Roll angle (radians, right-hand rule)
            pitch_camera: Pitch angle (radians); positive = camera tilted DOWN
            yaw_camera: Yaw angle (radians), positive CCW seen from above
            camera_intrinsics: Dictionary with keys 'fx', 'fy', 'cx', 'cy'
                               Required for pixel-to-3D conversion

        Camera Intrinsics Format:
            {
                'fx': focal length in x (pixels),
                'fy': focal length in y (pixels),
                'cx': principal point x (pixels),
                'cy': principal point y (pixels)
            }
        """
        # Store camera mounting configuration
        self.x_camera = x_camera
        self.y_camera = y_camera
        self.z_camera = z_camera
        self.roll_camera = roll_camera
        self.pitch_camera = pitch_camera
        self.yaw_camera = yaw_camera

        # Store camera intrinsics
        self.camera_intrinsics = camera_intrinsics

        # Build SE(3) transformation matrix: camera → vehicle,
        # operating on the "camera-in-vehicle-axes" frame (after remap).
        #
        # User convention: pitch_camera > 0 = camera tilted DOWN.
        # Right-hand rule: positive pitch = nose UP.
        # So we negate pitch when building the transform.
        self.T_vehicle_from_camera = se3_from_xyz_rpy(
            x_camera,
            y_camera,
            z_camera,
            roll_camera,
            -pitch_camera,
            yaw_camera,
        )

        # Optional inverse: vehicle → camera
        self.T_camera_from_vehicle = np.linalg.inv(self.T_vehicle_from_camera)

    # Internal axis remap

    def _remap_camera_axes_to_vehicle(self, cones_camera_3d: np.ndarray) -> np.ndarray:
        """
        Remap OpenCV camera coordinate axes to vehicle coordinate axes.

        OpenCV Camera Frame:
            X_cam = right
            Y_cam = down
            Z_cam = forward

        Vehicle Frame (Common Robotics Convention):
            X_veh = forward
            Y_veh = left
            Z_veh = up

        Solution: Apply a fixed rotation (just a matrix multiply) to remap the axes
        Fixed axis transformation:
            X_veh =  Z_cam
            Y_veh = -X_cam
            Z_veh = -Y_cam

        Args:
            cones_camera_3d: (N x 3) array of positions in OpenCV camera frame
                             Each row is [X_cam, Y_cam, Z_cam]

        Returns:
            (N x 3) array of positions in "camera-in-vehicle-axes" frame
            (same axes as vehicle, but still centered at the camera)

        Input: 3D points in camera axes (cones_camera_3d, N×3)
        Output: 3D points in vehicle-style axes, origin still at camera
        """
        cones_camera_3d = np.asarray(cones_camera_3d)
        R_remap = np.array([
            [0,  0,  1],   # X_veh = Z_cam
            [-1, 0,  0],   # Y_veh = -X_cam
            [0, -1,  0],   # Z_veh = -Y_cam
        ])
        return (R_remap @ cones_camera_3d.T).T

    # Pixel → 3D in camera (OpenCV) frame

    def pixel_to_camera_3d(
        self,
        pixel_u: int,
        pixel_v: int,
        depth_meters: float,
    ) -> np.ndarray:
        """
        Convert a single pixel coordinate + depth to 3D camera (OpenCV) frame.
        Problem: YOLACT gives you pixel coordinates of cones (u, v) and the depth map 
        gives you distance from camera. But pixels are in 2D, you need 3D points in camera space.

        So:
            - The camera captures 2D pixels (u, v) of a 3D point in the world
            - The pinhole model tells us how to go backwards: 
              from a 2D pixel and the distance to the object (depth), 
              find its 3D position relative to the camera.

        Solution (Stole the Math):
        Pinhole Camera Model:
            X_cam = (u - cx) * Z_cam / fx
            Y_cam = (v - cy) * Z_cam / fy
            Z_cam = depth

            What each variable means in the math:
            - (u,v) are pixel coordinates
            - (cx, cy) are principal point offsets from intrinsics
            - (fx, fy) = focal lengths in pixels
            - Z_cam = depth (how far the cone is from the camera along its forward axis)

        Args:
            pixel_u: Horizontal pixel coordinate (column, 0 = left)
            pixel_v: Vertical pixel coordinate (row, 0 = top)
            depth_meters: Distance to object in meters (from depth map)

        Returns:
            (3,) numpy array [X_cam, Y_cam, Z_cam] in OpenCV camera frame

        Raises:
            ValueError: If camera_intrinsics were not provided
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not provided. Must set in __init__().")

        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]

        z_cam = depth_meters
        x_cam = (pixel_u - cx) * z_cam / fx
        y_cam = (pixel_v - cy) * z_cam / fy

        return np.array([x_cam, y_cam, z_cam])

    def pixels_to_camera_3d_batch(
        self,
        pixels_u: np.ndarray,
        pixels_v: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """
        Convert multiple pixel coordinates + depths to 3D camera (OpenCV) coordinates.
        Instead of calling pixel_to_camera_3d in a loop, this vectorized function does all points in one shot faster and more efficient.
       
        Vectorized version of pixel_to_camera_3d.

        Args:
            pixels_u: (N,) array of horizontal pixel coordinates
            pixels_v: (N,) array of vertical pixel coordinates
            depths:   (N,) array of depths in meters

        Returns:
            (N x 3) array of 3D positions in camera (OpenCV, RDF: Right–Down–Forward) frame
            Each row is [X_cam, Y_cam, Z_cam]

        Raises:
            ValueError: If camera_intrinsics were not provided
        """
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not provided.")

        pixels_u = np.asarray(pixels_u) #where it appears in the camera image
        pixels_v = np.asarray(pixels_v)
        depths = np.asarray(depths) #how far it is from the camera

        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]

        x_cam = (pixels_u - cx) * depths / fx # side-to-side position relative to camera (right is +)
        y_cam = (pixels_v - cy) * depths / fy # vertical position relative to camera (down is +)
        z_cam = depths # distance forward from camera

        return np.stack([x_cam, y_cam, z_cam], axis=1) #Each row = [X_cam, Y_cam, Z_cam] in camera frame (Right–Down–Forward)

    # Camera (OpenCV) → Vehicle (3D and 2D)

    def camera_to_vehicle_3d(self, cones_camera_3d: np.ndarray) -> np.ndarray:
        """
        Transform 3D cone positions from OpenCV camera frame to vehicle frame (,X = forward, Y = left, Z = up).

        Steps:
        1. Remap axes (OpenCV camera → vehicle-like axes, origin at camera).
        2. Apply SE(3) transform camera→vehicle (camera mounting).

        Args:
            cones_camera_3d: (N x 3) array of positions in OpenCV camera frame

        Returns:
            (N x 3) array of positions in vehicle frame (Foward-Left-Up used for robotics)
        """
        cones_camera_3d = np.asarray(cones_camera_3d)

        # Step 1: remap coordinate axes
        cones_reoriented = self._remap_camera_axes_to_vehicle(cones_camera_3d)

        # Step 2: apply SE(3) camera→vehicle
        cones_vehicle_3d = transform_points_se3(
            self.T_vehicle_from_camera,
            cones_reoriented,
        )

        return cones_vehicle_3d

    def camera_to_vehicle_2d(self, cones_camera_3d: np.ndarray) -> np.ndarray:
        """
        Transform 3D cone positions from camera frame to 2D vehicle frame (ground plane).

        Complete pipeline:
        1. Remap axes (OpenCV camera → vehicle axes, origin at camera).
        2. Apply SE(3) transform (camera mounting) to get vehicle frame.
        3. Project to ground plane (discard Z coordinate).

        This assumes all cones are approximately on the ground (Z ≈ 0 in vehicle frame).

        Args:
            cones_camera_3d: (N x 3) array of positions in OpenCV camera frame

        Returns:
            (N x 2) array of positions in 2D vehicle frame (ground plane)
            Each row is [X_veh, Y_veh]:
                X_veh: forward distance from vehicle origin
                Y_veh: left distance from vehicle centerline
        """
        cones_vehicle_3d = self.camera_to_vehicle_3d(cones_camera_3d)
        cones_vehicle_2d = cones_vehicle_3d[:, :2]
        return cones_vehicle_2d

    # Debug info

    def info(self):
        """
        Print camera configuration and transforms for debugging.
        """
        print("Camera Position in Vehicle Frame:")
        print(f"  Forward (x): {self.x_camera:.2f} m")
        print(f"  Left   (y): {self.y_camera:.2f} m")
        print(f"  Up     (z): {self.z_camera:.2f} m")

        print("\nCamera Orientation (user convention):")
        print(f"  Roll : {np.degrees(self.roll_camera):.1f} deg")
        print(
            f"  Pitch: {np.degrees(self.pitch_camera):.1f} deg "
            f"{'(down)' if self.pitch_camera > 0 else '(up or level)'}"
        )
        print(f"  Yaw  : {np.degrees(self.yaw_camera):.1f} deg")

        if self.camera_intrinsics:
            print("\nCamera Intrinsics:")
            print(f"  fx: {self.camera_intrinsics['fx']:.1f} px")
            print(f"  fy: {self.camera_intrinsics['fy']:.1f} px")
            print(f"  cx: {self.camera_intrinsics['cx']:.1f} px")
            print(f"  cy: {self.camera_intrinsics['cy']:.1f} px")
        else:
            print("\nCamera Intrinsics: Not provided")

        print("\nT_vehicle_from_camera (4x4):")
        print(self.T_vehicle_from_camera)