# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:20:28 2024

@author: Vinayak
"""

import os
import cv2

input_image_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/eye/'
input_mask_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/mask/'
output_image_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/ResizedEye/'
output_mask_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/ResizedMask/'

target_size = (224, 224)

# Create output directories if they do not exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Check if input directories exist
if not os.path.exists(input_image_dir):
    print(f"Input image directory {input_image_dir} does not exist.")
if not os.path.exists(input_mask_dir):
    print(f"Input mask directory {input_mask_dir} does not exist.")

# Loop through image files and process them
for filename in os.listdir(input_image_dir):
    if filename.lower().endswith('.jpg'):
        print(f"Processing file: {filename}")
        
        # Process the image
        image_path = os.path.join(input_image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {filename}")
            continue
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        output_image_path = os.path.join(output_image_dir, filename)
        cv2.imwrite(output_image_path, resized_image)
        print(f"Resized image saved to {output_image_path}")
        
for mask_filename in os.listdir(input_mask_dir):        
        # Process the corresponding mask
        # Adjust case sensitivity for mask filenames
        # mask_filename = filename.replace('.JPG', '.png')  # Adjust based on your naming convention
        mask_path = os.path.join(input_mask_dir, mask_filename)
        if os.path.exists(mask_path):  # Check if mask file exists
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to read mask: {mask_filename}")
                continue
            resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            output_mask_path = os.path.join(output_mask_dir, mask_filename)
            cv2.imwrite(output_mask_path, resized_mask)
            print(f"Resized mask saved to {output_mask_path}")
        else:
            print(f"Mask file not found for image: {filename}")