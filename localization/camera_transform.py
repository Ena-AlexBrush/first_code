"""
Camera to Vehicle Frame Transformation

This module handles converting 3D cone positions from camera frame
to vehicle frame, then projecting to 2D for mapping.

Coordinate Frames:
- Camera: X=right, Y=down, Z=forward (camera convention)
- Vehicle: X=forward, Y=left, Z=up (robotics convention)
- Map: X=forward, Y=left (2D ground plane)
"""

import numpy as np
from localization.se3_utils import se3_from_xyz_rpy, transform_points_se3

class CameraToVehicleTransform:
    """
    Manages the transformation from camera frame to vehicle frame.
    
    Camera placement on vehicle:
    - Mounted at height z_camera above ground
    - Offset x_camera forward from vehicle center
    - Possibly tilted down by pitch_camera to see the track
    """
    
    def __init__(self, 
                 x_camera: float = 0.5,      # meters forward from vehicle center
                 y_camera: float = 0.0,      # meters left from vehicle centerline
                 z_camera: float = 0.3,      # meters above ground
                 roll_camera: float = 0.0,   # radians (usually 0)
                 pitch_camera: float = 0.0,  # radians (positive = tilted down)
                 yaw_camera: float = 0.0):   # radians (usually 0, camera faces forward)
        
        """
        Initialize the camera-to-vehicle transformation.
        
        Args:
            x_camera: Forward distance from vehicle origin (meters)
            y_camera: Left distance from vehicle centerline (meters)  
            z_camera: Height above ground (meters)
            roll_camera: Roll angle (radians)
            pitch_camera: Pitch angle - positive means tilted down (radians)
            yaw_camera: Yaw angle (radians)
        """
        self.x_camera = x_camera
        self.y_camera = y_camera
        self.z_camera = z_camera
        self.roll_camera = roll_camera
        self.pitch_camera = pitch_camera
        self.yaw_camera = yaw_camera
        
        # Compute the transform matrix: Vehicle frame -> Camera frame
        # Invert this to go Camera -> Vehicle
        
        self.T_vehicle_to_camera = se3_from_xyz_rpy(
            x_camera, y_camera, z_camera,
            roll_camera, pitch_camera, yaw_camera
        )
        
        # Invert to get Camera → Vehicle
        self.T_camera_to_vehicle = np.linalg.inv(self.T_vehicle_to_camera)
    
    def camera_to_vehicle_3d(self, cones_camera_3d: np.ndarray) -> np.ndarray:
        """
        Transform 3D cone positions from camera frame to vehicle frame.
        
        Args:
            cones_camera_3d: (N x 3) array of [x, y, z] positions in camera frame
            
        Returns:
            (N x 3) array of [x, y, z] positions in vehicle frame
        """
        return transform_points_se3(self.T_camera_to_vehicle, cones_camera_3d)
    
    def camera_to_vehicle_2d(self, cones_camera_3d: np.ndarray) -> np.ndarray:
        """
        Transform 3D cone positions from camera frame to 2D vehicle frame.
        
        This is the key function that goes from 3D to 2D!
        
        Process:
        1. Transform from camera 3D to vehicle 3D
        2. Project to ground plane by taking only (x, y) coordinates
        3. Ignore z-coordinate (assumes cones are on the ground)
        
        Args:
            cones_camera_3d: (N x 3) array of [x, y, z] positions in camera frame
            
        Returns:
            (N x 2) array of [x, y] positions in 2D vehicle frame (ground plane)
        """
        # Transform to 3D vehicle frame
        cones_vehicle_3d = self.camera_to_vehicle_3d(cones_camera_3d)
        
        # Project to 2D by taking only x and y (discard z)
        # This assumes cones are on the ground plane
        cones_vehicle_2d = cones_vehicle_3d[:, :2]  # Shape: (N, 2)
        
        return cones_vehicle_2d
    
    def info(self):
        """Print information about the camera setup."""
        print("Camera Position in Vehicle Frame:")
        print(f"  Forward (x): {self.x_camera:.2f} m")
        print(f"  Left (y): {self.y_camera:.2f} m")
        print(f"  Up (z): {self.z_camera:.2f} m")
        print(f"\nCamera Orientation:")
        print(f"  Roll: {np.degrees(self.roll_camera):.1f}°")
        print(f"  Pitch: {np.degrees(self.pitch_camera):.1f}° {'(down)' if self.pitch_camera > 0 else '(up)'}")
        print(f"  Yaw: {np.degrees(self.yaw_camera):.1f}°")


def create_fake_camera_detections(num_cones: int = 5, 
                                   distance_range: tuple = (3.0, 10.0)) -> np.ndarray:
    """
    Generate fake 3D cone positions in camera frame for testing.
    Simulates what you'd get from YOLACT + Depth Estimation.
    Camera frame: X=right, Y=down, Z=forward
    
    Args:
        num_cones: Number of fake cones to generate
        distance_range: (min, max) distance in meters
        
    Returns:
        (N x 3) array of fake cone positions in camera frame
    """
    np.random.seed(42)  # For reproducibility
    
    cones = [] # List to hold cone pos
    for i in range(num_cones):
        # Distance forward (Z in camera frame)
        z = np.random.uniform(distance_range[0], distance_range[1])
        
        # Lateral offset (X in camera frame) - cones can be left or right
        x = np.random.uniform(-3.0, 3.0)
        
        # Height (Y in camera frame) - assume on ground, so small positive Y
        # (Y is down in camera frame, so ground is at Y ≈ camera_height)
        y = np.random.uniform(0.2, 0.4)  # Small variation
        
        cones.append([x, y, z])
    
    return np.array(cones)


# Example usage
# Need to add more cones
if __name__ == "__main__":
    print("=" * 60)
    print("Camera to Vehicle Transformation Demo")
    print("=" * 60)
    
    # Create transformer with typical Formula Student camera setup
    transformer = CameraToVehicleTransform(
        x_camera=0.5,    # 50cm forward from vehicle center
        y_camera=0.0,    # Centered on vehicle
        z_camera=0.3,    # 30cm above ground
        pitch_camera=np.radians(10)  # Tilted down 10 degrees to see track
    )
    
    transformer.info()
    
    # Generate fake cone detections in camera frame
    print("\n" + "=" * 60)
    print("Fake Camera Detections (Camera Frame)")
    print("=" * 60)
    cones_camera = create_fake_camera_detections(num_cones=5)
    print("Camera frame: X=right, Y=down, Z=forward")
    print(cones_camera)
    
    # Transform to 2D vehicle frame
    print("\n" + "=" * 60)
    print("Transformed to Vehicle Frame (2D)")
    print("=" * 60)
    cones_vehicle_2d = transformer.camera_to_vehicle_2d(cones_camera)
    print("Vehicle frame: X=forward, Y=left")
    print(cones_vehicle_2d)
    
    print("\n✅ Success! We went from 3D camera data to 2D vehicle data!")
    print("These 2D positions can now go into your existing mapping pipeline.")

#Newer code:

# """
# Camera to Vehicle Frame Transformation

# Converts 3D cone positions from camera frame → vehicle frame → 2D map

# COORDINATE FRAMES:
# - Camera: X=right, Y=down, Z=forward (standard camera convention)
# - Vehicle: X=forward, Y=left, Z=up (robotics convention)
# - Map: X=forward, Y=left (2D ground plane only)

# HOW 3D BECOMES 2D:
# 1. Transform camera 3D → vehicle 3D (using SE(3))
# 2. Project to ground by taking only (x, y), dropping z
# 3. Assumption: All cones are on the ground (z ≈ 0)
# """

# import numpy as np
# from localization.se3_utils import se3_from_xyz_rpy, transform_points_se3  # FIXED TYPO!


# class CameraToVehicleTransform:
#     """
#     Handles transformation from camera frame to vehicle frame.
    
#     Camera mount configuration:
#     - x_camera: Forward offset from vehicle center
#     - y_camera: Left offset from centerline  
#     - z_camera: Height above ground
#     - pitch_camera: Tilt angle (positive = looking down)
#     """
    
#     def __init__(self, 
#                  x_camera: float = 0.5,      # Forward (meters)
#                  y_camera: float = 0.0,      # Left (meters)
#                  z_camera: float = 0.3,      # Up (meters)
#                  roll_camera: float = 0.0,   # Roll (radians)
#                  pitch_camera: float = 0.0,  # Pitch (radians, + = down)
#                  yaw_camera: float = 0.0):   # Yaw (radians)
#         """
#         Initialize camera-to-vehicle transformation.
        
#         NOTE: Measure these values on your actual car!
#         """
#         # Store camera position/orientation
#         self.x_camera = x_camera
#         self.y_camera = y_camera
#         self.z_camera = z_camera
#         self.roll_camera = roll_camera
#         self.pitch_camera = pitch_camera
#         self.yaw_camera = yaw_camera
        
#         # Build transformation matrix: Vehicle → Camera
#         self.T_vehicle_to_camera = se3_from_xyz_rpy(
#             x_camera, y_camera, z_camera,
#             roll_camera, pitch_camera, yaw_camera
#         )
        
#         # Invert to get: Camera → Vehicle
#         self.T_camera_to_vehicle = np.linalg.inv(self.T_vehicle_to_camera)
    
#     def camera_to_vehicle_3d(self, cones_camera_3d: np.ndarray) -> np.ndarray:
#         """
#         Transform 3D cone positions from camera → vehicle frame.
        
#         Args:
#             cones_camera_3d: (N x 3) array of [x, y, z] in camera frame
        
#         Returns:
#             (N x 3) array of [x, y, z] in vehicle frame
#         """
#         return transform_points_se3(self.T_camera_to_vehicle, cones_camera_3d)
    
#     def camera_to_vehicle_2d(self, cones_camera_3d: np.ndarray) -> np.ndarray:
#         """
#         Transform 3D cones from camera → 2D vehicle frame (for mapping).
        
#         This is the KEY function! Goes from 3D → 2D in two steps:
#         1. Transform to vehicle 3D coordinates
#         2. Drop z-coordinate (project to ground plane)
        
#         Args:
#             cones_camera_3d: (N x 3) array in camera frame
        
#         Returns:
#             (N x 2) array of [x, y] in 2D vehicle frame (ground plane)
#         """
#         # Step 1: Transform to 3D vehicle frame
#         cones_vehicle_3d = self.camera_to_vehicle_3d(cones_camera_3d)
        
#         # Step 2: Project to 2D by taking only x and y
#         # (Assumes cones are on ground, so we ignore z)
#         cones_vehicle_2d = cones_vehicle_3d[:, :2]  # Just take first 2 columns
        
#         return cones_vehicle_2d
    
#     def info(self):
#         """Print camera configuration info."""
#         print("Camera Position in Vehicle Frame:")
#         print(f"  Forward (x): {self.x_camera:.2f} m")
#         print(f"  Left (y):    {self.y_camera:.2f} m")
#         print(f"  Up (z):      {self.z_camera:.2f} m")
#         print(f"\nCamera Orientation:")
#         print(f"  Roll:  {np.degrees(self.roll_camera):.1f}°")
#         print(f"  Pitch: {np.degrees(self.pitch_camera):.1f}° {'(down)' if self.pitch_camera > 0 else '(up)'}")
#         print(f"  Yaw:   {np.degrees(self.yaw_camera):.1f}°")


# # ============================================================================
# # Testing Helper - Generates Fake Cones
# # ============================================================================

# def create_fake_camera_detections(num_cones: int = 5, 
#                                    distance_range: tuple = (3.0, 10.0)) -> np.ndarray:
#     """
#     Generate fake 3D cone positions in camera frame (for testing).
    
#     Simulates what YOLACT + Depth would give you.
#     Camera frame: X=right, Y=down, Z=forward
    
#     Args:
#         num_cones: Number of fake cones
#         distance_range: (min, max) distance in meters
    
#     Returns:
#         (N x 3) array of fake cone positions
#     """
#     np.random.seed(42)  # Reproducible results
    
#     cones = []
#     for i in range(num_cones):
#         # Z = distance forward
#         z = np.random.uniform(distance_range[0], distance_range[1])
        
#         # X = left/right offset
#         x = np.random.uniform(-3.0, 3.0)
        
#         # Y = height (small, since cones are on ground)
#         y = np.random.uniform(0.2, 0.4)
        
#         cones.append([x, y, z])
    
#     return np.array(cones)


# # ============================================================================
# # Testing
# # ============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("CAMERA TRANSFORMATION TEST")
#     print("=" * 70)
    
#     # Create transformer with typical Formula Student setup
#     transformer = CameraToVehicleTransform(
#         x_camera=0.5,                   # 50cm forward
#         y_camera=0.0,                   # Centered
#         z_camera=0.3,                   # 30cm high
#         pitch_camera=np.radians(10)     # 10° down
#     )
    
#     transformer.info()
    
#     # Generate fake detections
#     print("\n" + "=" * 70)
#     print("FAKE CONE DETECTIONS (Camera Frame)")
#     print("=" * 70)
#     print("Camera: X=right, Y=down, Z=forward")
    
#     cones_camera = create_fake_camera_detections(num_cones=5)
#     print(cones_camera)
    
#     # Transform to 2D vehicle frame
#     print("\n" + "=" * 70)
#     print("TRANSFORMED TO VEHICLE FRAME (2D)")
#     print("=" * 70)
#     print("Vehicle: X=forward, Y=left")
    
#     cones_vehicle_2d = transformer.camera_to_vehicle_2d(cones_camera)
#     print(cones_vehicle_2d)
    
#     print("\n✅ Success! 3D camera data → 2D vehicle data")
#     print("   Ready to map into global coordinates!")