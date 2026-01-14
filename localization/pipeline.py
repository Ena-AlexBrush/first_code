"""
Main Localization Pipeline

Connects all modules together:
    IMU → Vehicle Pose
    YOLACT + Depth → 3D Cones in Camera Frame
    Camera Transform → 2D Cones in Vehicle Frame
    Pose + Cones → Global Cone Map
"""

import numpy as np
import polars as pl

from localization.imu_integration import load_imu_data, estimate_pose_from_imu
from localization.yolact_integration import parse_yolact_detections, filter_low_confidence_detections
from localization.depth_integration import load_depth_map
from localization.camera_transform import CameraToVehicleTransform
from localization.cone_mapping import cones_vehicle_to_global


class LocalizationPipeline:
    """
    Main pipeline that processes sensor data to produce global cone map.
    """
    
    def __init__(self, camera_transform: CameraToVehicleTransform):
        """
        Initialize pipeline with camera configuration.
        
        Args:
            camera_transform: CameraToVehicleTransform with your camera setup
        """
        self.camera_transform = camera_transform
        self.pose_df = None
    
    def process_imu(self, imu_csv_path: str, v_forward: float = 3.0) -> pl.DataFrame: #Assumed 3 m/s forward velocity
        """
        Step 1: Process IMU data to get vehicle pose over time.
        
        Args:
            imu_csv_path: Path to IMU CSV file
            v_forward: Assumed forward velocity (m/s)
            
        Returns:
            DataFrame with columns: time_s, x, y, theta
        """
        print("Step 1: Processing IMU data...")
        
        # Load raw IMU data
        imu_df = load_imu_data(imu_csv_path)
        
        # Estimate pose (this will be replaced by Kimera later)
        self.pose_df = estimate_pose_from_imu(imu_df, v_forward)
        
        print(f"Estimated {len(self.pose_df)} vehicle poses")
        return self.pose_df
    
    def process_detections(self, 
                          yolact_output: dict,
                          depth_map_data,
                          camera_intrinsics: dict) -> np.ndarray:
        """
        Step 2: Process YOLACT + Depth to get 3D cone positions.
        
        Args:
            yolact_output: Dictionary from YOLACT team
            depth_map_data: Depth array from Depth team
            camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy'
            
        Returns:
            (N x 3) array of cone positions in camera frame [x, y, z]
        """
        print("Step 2: Processing cone detections...")
        
        # Parse YOLACT detections
        detections_df = parse_yolact_detections(yolact_output)
        detections_df = filter_low_confidence_detections(detections_df, min_confidence=0.7)
        
        # Load depth map
        depth_map = load_depth_map(depth_map_data)
        
        # Convert pixel + depth to 3D camera coordinates
        cones_camera_3d = []
        for row in detections_df.iter_rows(named=True):
            u = row['pixel_u']
            v = row['pixel_v']
            depth = depth_map.get_depth_at_pixel(u, v)
            
            # Use pinhole camera model to backproject
            x_cam = (u - camera_intrinsics['cx']) * depth / camera_intrinsics['fx'] # x = (u - cx) * z / fx is the math
            y_cam = (v - camera_intrinsics['cy']) * depth / camera_intrinsics['fy'] # y = (v - cy) * z / fy is the math
            z_cam = depth
            
            cones_camera_3d.append([x_cam, y_cam, z_cam])
        
        cones_camera_3d = np.array(cones_camera_3d)
        print(f"   ✓ Detected {len(cones_camera_3d)} cones in 3D")
        
        return cones_camera_3d
    
    def transform_to_vehicle_frame(self, cones_camera_3d: np.ndarray) -> np.ndarray:
        """
        Step 3: Transform cones from camera frame to 2D vehicle frame.
        
        Args:
            cones_camera_3d: (N x 3) array in camera frame
            
        Returns:
            (N x 2) array in vehicle frame (ground plane)
        """
        print("Step 3: Transforming to vehicle frame...")
        
        cones_vehicle_2d = self.camera_transform.camera_to_vehicle_2d(cones_camera_3d)
        
        print(f"   ✓ Transformed to 2D vehicle frame")
        return cones_vehicle_2d
    
    def map_to_global_frame(self, 
                           cones_vehicle_2d: np.ndarray,
                           sample_every: int = 1) -> pl.DataFrame:
        """
        Step 4: Map cones from vehicle frame to global frame.
        
        Args:
            cones_vehicle_2d: (N x 2) array in vehicle frame
            sample_every: Use every Nth pose (for performance)
            
        Returns:
            DataFrame with global cone positions
        """
        print("🌍 Step 4: Mapping to global frame...")
        
        if self.pose_df is None:
            raise RuntimeError("Must run process_imu() first!")
        
        global_cones_df = cones_vehicle_to_global(
            self.pose_df,
            cones_vehicle_2d,
            sample_every=sample_every
        )
        
        print(f"   ✓ Created global map with {len(global_cones_df)} observations")
        return global_cones_df
    
    def run_full_pipeline(self,
                         imu_csv_path: str,
                         yolact_output: dict,
                         depth_map_data,
                         camera_intrinsics: dict) -> pl.DataFrame:
        """
        Run complete pipeline from raw sensor data to global map.
        
        Args:
            imu_csv_path: Path to IMU data
            yolact_output: YOLACT detections
            depth_map_data: Depth map
            camera_intrinsics: Camera calibration
            
        Returns:
            Global cone map DataFrame
        """
        print("=" * 70)
        print("RUNNING COMPLETE LOCALIZATION PIPELINE")
        print("=" * 70)
        
        # Step 1: IMU → Pose
        self.process_imu(imu_csv_path)
        
        # Step 2: YOLACT + Depth → 3D Cones
        cones_camera_3d = self.process_detections(
            yolact_output, 
            depth_map_data, 
            camera_intrinsics
        )
        
        # Step 3: Camera → Vehicle (2D)
        cones_vehicle_2d = self.transform_to_vehicle_frame(cones_camera_3d)
        
        # Step 4: Vehicle → Global
        global_map = self.map_to_global_frame(cones_vehicle_2d)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        
        return global_map