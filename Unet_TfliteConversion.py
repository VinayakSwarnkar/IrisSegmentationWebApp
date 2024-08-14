# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:40:10 2024

@author: Vinayak
"""

import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Unet_IrisSegmentation_model.h5')

# Create a TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optionally, set optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Unet_IrisSegmentation_model.tflite', 'wb') as f:
    f.write(tflite_model)