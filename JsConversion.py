# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:14:51 2024

@author: Vinayak
"""

import tensorflow as tf
import tensorflowjs as tfjs

# Load your TensorFlow Lite model
model = tf.keras.models.load_model('Unet_IrisSegmentation_model.tflite', custom_objects=None, compile=False)

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'JsConvertedModel')