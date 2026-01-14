#NOTE: MAY NO LONGER BE NEEDED
"""
Pose Estimation Module

This module estimates the vehicle's position (x, y, theta) over time.
Currently uses simple IMU dead-reckoning. Later, swap in Kimera-VIO here.

INPUT:  IMU data (CSV from IMU team)
OUTPUT: Vehicle poses over time (x, y, theta)
"""

import polars as pl


def estimate_poses_from_imu(imu_csv_path: str, v_forward: float = 3.0) -> pl.DataFrame:
    """
    Estimate vehicle poses using simple IMU dead-reckoning.
    
    NOTE: This is a placeholder Kimera-VIO will replace this function later.
          The important part: the OUTPUT FORMAT stays the same!
    
    Args:
        imu_csv_path: Path to IMU CSV file (from IMU team)
        v_forward: Constant forward velocity in m/s (placeholder)
    
    Returns:
        DataFrame with columns:
            - time_s: Timestamp in seconds
            - x: Position in meters (forward)
            - y: Position in meters (left)
            - theta: Heading in radians
    """
    
    # STEP 1: Load raw IMU data
    df = pl.read_csv(imu_csv_path)
    
    # STEP 2: Extract needed columns and rename
    imu_df = df.select([
        pl.col("General.Time (millis)").alias("time_ms"),
        pl.col("ISM330.Accel X (milli-g)").alias("ax_mg"),
        pl.col("ISM330.Accel Y (milli-g)").alias("ay_mg"),
        pl.col("ISM330.Gyro Z (milli-dps)").alias("gyro_z_mdps"),
    ])
    
    # STEP 3: Convert time to seconds
    imu_df = imu_df.sort("time_ms").with_columns(
        (pl.col("time_ms") / 1000).alias("time_s")
    )
    
    # STEP 4: Convert gyro to rad/s (from milli-deg/s)
    # 57295.8 = 1000 * (180 / pi) conversion factor
    imu_df = imu_df.with_columns(
        (pl.col("gyro_z_mdps") / 57295.8).alias("gyro_z_rads")
    )
    
    # STEP 5: Calculate time step (dt)
    imu_df = imu_df.with_columns(
        pl.col("time_s").diff().fill_null(0.0).alias("dt")
    )
    
    # STEP 6: Integrate gyro to get heading (theta)
    imu_df = imu_df.with_columns(
        (pl.col("gyro_z_rads") * pl.col("dt")).alias("dtheta")
    ).with_columns(
        pl.col("dtheta").cum_sum().alias("theta")
    )
    
    # STEP 7: Calculate forward distance traveled each step
    imu_df = imu_df.with_columns(
        (v_forward * pl.col("dt")).alias("ds")
    )
    
    # STEP 8: Project forward motion into world coordinates
    imu_df = imu_df.with_columns(
        (pl.col("ds") * pl.col("theta").cos()).alias("dx_w"),
        (pl.col("ds") * pl.col("theta").sin()).alias("dy_w")
    )
    
    # STEP 9: Integrate to get absolute position
    imu_df = imu_df.with_columns(
        pl.col("dx_w").cum_sum().alias("x"),
        pl.col("dy_w").cum_sum().alias("y")
    )
    
    # STEP 10: Return clean pose DataFrame
    # This format MUST stay the same when swaped in Kimera!
    return imu_df.select(["time_s", "x", "y", "theta"])


# PLACEHOLDER FOR KIMERA-VIO


def estimate_poses_with_kimera(imu_data_path: str, 
                               camera_video_path: str,
                               config_path: str) -> pl.DataFrame:
    """
    Estimate vehicle poses using Kimera-VIO (visual-inertial odometry).
    
    TODO: Implement this after Parker finishes Kimera build (Jan 17).
    
    Args:
        imu_data_path: Path to IMU data
        camera_video_path: Path to camera video
        config_path: Path to Kimera config file
    
    Returns:
        DataFrame with SAME format as estimate_poses_from_imu():
            - time_s, x, y, theta
    
    The key: Output format is identical! Just swap which function you call.
    """
    raise NotImplementedError("Waiting for Parker's Kimera integration (Jan 17)")
    
    # When Parker is done, this will look like:
    # from kimera_vio import KimeraVIO
    # kimera = KimeraVIO(config_path)
    # ... process frames ...
    # return pl.DataFrame with time_s, x, y, theta

# Testing

if __name__ == "__main__":
    # Test with IMU data
    pose_df = estimate_poses_from_imu("data/Rectangle2x.csv", v_forward=3.0)
    
    print("=" * 60)
    print("POSE ESTIMATION TEST")
    print("=" * 60)
    print(f"\nGenerated {len(pose_df)} poses")
    print(f"Time range: {pose_df['time_s'].min():.2f}s to {pose_df['time_s'].max():.2f}s")
    print("\nFirst few poses:")
    print(pose_df.head())
    print("\nLast few poses:")
    print(pose_df.tail())