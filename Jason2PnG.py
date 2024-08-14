# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:56:24 2024

@author: Vinayak
"""

import os
import base64
import numpy as np
import cv2
import json
from labelme import utils

input_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/eye/'
output_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/mask/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(input_dir, filename)
        with open(json_path) as f:
            data = json.load(f)
        
        imageData = data.get("imageData")
        if not imageData:
            imagePath = os.path.join(input_dir, data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = base64.b64encode(f.read()).decode('utf-8')
        
        img = utils.img_b64_to_arr(imageData)
        
        # Define the label name you used in LabelMe
        label_name_to_value = {'_background_': 0, 'IRIS': 1}
        
        # Convert shapes to label (mask)
        lbl, lbl_names = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

        print(f"Label shape: {lbl.shape}, unique values: {np.unique(lbl)}")
        
        # Create mask
        mask = (lbl == label_name_to_value['IRIS']).astype(np.uint8)  # Use the value for 'IRIS'
        print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        
        mask_path = os.path.join(output_dir, filename.replace('.json', '_mask.png'))
        cv2.imwrite(mask_path, mask * 255)  # Save mask as binary image