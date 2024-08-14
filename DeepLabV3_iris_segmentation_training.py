import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

# Define paths
image_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/ResizedEye/'
mask_dir = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Dataset/ResizedMask/'

def load_data(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.JPG')])
    mask_paths = sorted([os.path.join(mask_dir, f.replace('.JPG', '_mask.png')) for f in os.listdir(image_dir) if f.endswith('.JPG')])

    images = []
    masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (224, 224))
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)
    
    images = np.array(images) / 255.0
    masks = np.array(masks)
    
    return images, masks

def create_deeplabv3_model(input_shape, num_classes):
    # Load DeepLabV3 model from TensorFlow Hub
    hub_url = "https://tfhub.dev/google/deeplabv3/1"  # Ensure this is the correct URL for segmentation
    base_model = hub.KerasLayer(hub_url, input_shape=input_shape, trainable=False)
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = UpSampling2D(size=(4, 4))(x)  # Upsample to original size
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load data
images, masks = load_data(image_dir, mask_dir)

# Debugging information
print(f'Loaded images: {len(images)}')
print(f'Loaded masks: {len(masks)}')

# Check shapes
if len(images) == 0 or len(masks) == 0:
    raise ValueError('No images or masks were loaded. Check file paths and data loading.')

# Split data
X_train_full, X_test, Y_train_full, Y_test = train_test_split(images, masks, test_size=0.20, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.20, random_state=42)

# Debugging information
print(f'Training images: {X_train.shape}')
print(f'Validation images: {X_val.shape}')
print(f'Test images: {X_test.shape}')

# Create DeepLabv3 model
input_shape = (224, 224, 3)
num_classes = 2  # For binary segmentation, adjust as needed
model = create_deeplabv3_model(input_shape, num_classes)

# Define callbacks
tensorboard_callback = TensorBoard(log_dir='./logs')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    batch_size=16,
    epochs=50,
    validation_data=(X_val, Y_val),
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
