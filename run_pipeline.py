# run_localization.py
"""
Main Entry Point for Localization System

Run this file to execute the complete pipeline.
Configure your camera and data paths below.
"""

import numpy as np
from localization.camera_transform import CameraToVehicleTransform
from localization.pipeline import LocalizationPipeline


def main():
    """
    Main function - configure and run pipeline.
    """
    
    # CONFIGURATION
    
    # Camera mounting configuration (MEASURE THESE ON THE CAR!)
    camera_config = {
        'x_camera': 0.5,                    # 50cm forward from vehicle center
        'y_camera': 0.0,                    # Centered on vehicle
        'z_camera': 0.3,                    # 30cm above ground
        'pitch_camera': np.radians(10),     # Tilted down 10 degrees
        'roll_camera': 0.0,
        'yaw_camera': 0.0,
    }
    
    # Data paths
    imu_path = "data/Rectangle2x.csv"
    
    # Camera intrinsics (WILL COME FROM PARKER'S CALIBRATION)
    # These are placeholder values
    camera_intrinsics = {
        'fx': 600.0,  # Focal length x (pixels)
        'fy': 600.0,  # Focal length y (pixels)
        'cx': 320.0,  # Principal point x (pixels)
        'cy': 240.0,  # Principal point y (pixels)
    }
    
    # ========================================================================
    # SETUP PIPELINE
    # ========================================================================
    
    print("Setting up localization pipeline...")
    
    # Initialize camera transform
    camera_transform = CameraToVehicleTransform(
        x_camera=camera_config['x_camera'],
        y_camera=camera_config['y_camera'],
        z_camera=camera_config['z_camera'],
        roll_camera=camera_config['roll_camera'],
        pitch_camera=camera_config['pitch_camera'],
        yaw_camera=camera_config['yaw_camera'],
    )
    
    # Create pipeline
    pipeline = LocalizationPipeline(camera_transform)
    
    # ========================================================================
    # RUN WITH IMU ONLY (FOR NOW)
    # ========================================================================
    
    print("\n🔹 Running with IMU data only (YOLACT + Depth not yet integrated)\n")
    
    # Process just the IMU to get vehicle trajectory
    pose_df = pipeline.process_imu(imu_path, v_forward=3.0)
    
    print("\n📊 Vehicle Trajectory:")
    print(pose_df.head(10))
    
    print(f"\n✅ Successfully estimated {len(pose_df)} vehicle poses")
    print(f"   Start position: ({pose_df['x'][0]:.2f}, {pose_df['y'][0]:.2f})")
    print(f"   End position: ({pose_df['x'][-1]:.2f}, {pose_df['y'][-1]:.2f})")
    
    # ========================================================================
    # FUTURE: RUN WITH FULL DATA (AFTER YOLACT + DEPTH INTEGRATED)
    # ========================================================================
    
    # Uncomment this when you have real YOLACT and Depth data:
    
    # yolact_output = {
    #     'classes': [...],
    #     'confidence_scores': [...],
    #     'bounding_boxes': [...],
    # }
    # depth_map_data = np.load("data/depth_map.npy")
    #
    # global_map = pipeline.run_full_pipeline(
    #     imu_csv_path=imu_path,
    #     yolact_output=yolact_output,
    #     depth_map_data=depth_map_data,
    #     camera_intrinsics=camera_intrinsics
    # )
    #
    # print("\n🗺️  Global Cone Map:")
    # print(global_map.head(20))
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Measure camera position on your car")
    print("2. Update camera_config values above")
    print("3. Wait for YOLACT + Depth data")
    print("4. Uncomment full pipeline section")
    print("5. Run again with complete data!")


if __name__ == "__main__":
    main()