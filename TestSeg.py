# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:29:32 2024

@author: Vinayak
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np


# Load a pre-trained U-Net model
model = load_model('unet_iris_segmentation.h5')

# Load and preprocess the image
# Load the image
image = cv2.imread('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/eye/1-543.JPG')
image_resized = cv2.resize(image, (128, 128))  # Resize to model input size
image_resized = image_resized / 255.0  # Normalize
image_resized = np.expand_dims(image_resized, axis=0)

# Predict the segmentation mask
mask = model.predict(image_resized)[0]
mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask

# Post-process the mask
mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize to original image size
mask_colored = cv2.applyColorMap(mask_resized * 255, cv2.COLORMAP_JET)  # Apply color map

# Overlay the mask on the original image
segmented_image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

cv2.imshow('Iris Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()