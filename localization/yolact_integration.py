"""
YOLACT Detection Integration Module

Input Expected from YOLACT Team:
    - classes: Array of integers [0, 1, 2, 3] representing cone types
        - 255 = none, 0 = blue, 1 = yellow, 2 = sOrange, 3 = bOrange
        - Pixel % 4 = cone type
        - Pixel // 4 = cone #
    - confidence_scores: Array of floats [0.0 to 1.0] for detection confidence
    - bounding_boxes: Array of [x1, y1, x2, y2] pixel coordinates
    - masks: Array of pixel masks (optional, may not need??)

Output:
    - DataFrame with pixel locations of detected cones
"""

import polars as pl
import numpy as np


def parse_yolact_detections(yolact_output: dict) -> pl.DataFrame:
    """
    Convert YOLACT output into structured cone detection data.
    
    Args:
        yolact_output: Dictionary containing:
            {
                'classes': [0, 1, 2, 3, ...],              # Cone types
                'confidence_scores': [0.95, 0.87, ...],    # Detection confidence
                'bounding_boxes': [[x1,y1,x2,y2], ...]     # Pixel coordinates
            }
            
    Returns:
        DataFrame with columns:
            - cone_id: Unique ID for this detection
            - cone_type: Type of cone (0-3)
            - confidence: Detection confidence score
            - pixel_u: Horizontal center pixel (column)
            - pixel_v: Vertical center pixel (row)
    """
    detections = []
    
    # Extract each cone detection
    for i in range(len(yolact_output['classes'])):
        bbox = yolact_output['bounding_boxes'][i]
        
        # Calculate center of bounding box
        # u = column (horizontal), v = row (vertical)
        center_u = int((bbox[0] + bbox[2]) / 2)
        center_v = int((bbox[1] + bbox[3]) / 2)
        
        detections.append({
            'cone_id': i,
            'cone_type': yolact_output['classes'][i],
            'confidence': yolact_output['confidence_scores'][i],
            'pixel_u': center_u,
            'pixel_v': center_v,
        })
    
    return pl.DataFrame(detections)


def filter_low_confidence_detections(detections_df: pl.DataFrame, 
                                     min_confidence: float = 0.7) -> pl.DataFrame:
    """
    Remove detections with low confidence scores.
    
    Args:
        detections_df: Output from parse_yolact_detections()
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        Filtered DataFrame
    """
    return detections_df.filter(pl.col("confidence") >= min_confidence)