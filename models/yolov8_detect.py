#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Anmol
#
# Created:     17-04-2024
# Copyright:   (c) Anmol 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from PIL import Image
import io

def detect_objects(image_data):
    # Load YOLOv5 model
    model = attempt_load("best.pt", map_location=torch.device('cpu'))

    # Process image
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')

    # Perform inference
    results = model(img)

    # Post-processing
    # Example: Non-maximum suppression
    results = non_max_suppression(results, conf_thres=0.4)

    # Format results
    detected_objects = []
    for detections in results:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            detected_objects.append({
                'class': int(cls_pred),
                'confidence': float(cls_conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

    return detected_objects
