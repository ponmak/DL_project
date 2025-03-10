import supervision as sv
import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO

def detection(image, model):
    """
    Run object detection on an image using the YOLO model.
    
    Args:
        image: Input image for detection
        model: YOLO model instance
        
    Returns:
        Detections object containing detection results
    """
    # Handle different image input types
    if isinstance(image, str) and os.path.isfile(image):
        # If image is a file path
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif hasattr(image, 'getvalue'):
        # If image is a Streamlit UploadedFile or camera input
        file_bytes = np.asarray(bytearray(image.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already a numpy array
        img = image
    
    # Run detection
    results = model(img)
    
    # Convert to supervision Detections format
    detections = sv.Detections.from_ultralytics(results[0]).with_nms()

    # Annotate image
    box_annotator = sv.BoxAnnotator(thickness=5, color=sv.Color.GREEN)
    label_annotator = sv.LabelAnnotator()
    annotated_image = img.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    return annotated_image

