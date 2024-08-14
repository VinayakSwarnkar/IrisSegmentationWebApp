# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:06:10 2024

@author: Vinayak
"""
import numpy as np
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (224, 224))
            mask = (mask > 0).astype(np.float32)  # Binarize the mask
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)
    
    images = np.array(images) / 255.0
    masks = np.array(masks)
    
    return images, masks, image_paths, mask_paths

def create_unet_model(input_shape):
    inputs = Input(input_shape)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load data
images, masks, image_paths, mask_paths = load_data(image_dir, mask_dir)

# Split data and save indices
X_train_full, X_test, Y_train_full, Y_test, train_full_paths, test_paths = train_test_split(
    images, masks, list(zip(image_paths, mask_paths)), test_size=0.15, random_state=42
)
X_train, X_val, Y_train, Y_val, train_paths, val_paths = train_test_split(
    X_train_full, Y_train_full, train_full_paths, test_size=0.20, random_state=42
)

# Save paths to files
with open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/train_paths.txt', 'w') as f:
    for img_path, mask_path in train_paths:
        f.write(f"{img_path},{mask_path}\n")

with open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/val_paths.txt', 'w') as f:
    for img_path, mask_path in val_paths:
        f.write(f"{img_path},{mask_path}\n")

with open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/test_paths.txt', 'w') as f:
    for img_path, mask_path in test_paths:
        f.write(f"{img_path},{mask_path}\n")

# Create U-Net model
input_shape = (224, 224, 3)
model = create_unet_model(input_shape)

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
train_loss, train_accuracy = model.evaluate(X_train, Y_train)
val_loss, val_accuracy = model.evaluate(X_val, Y_val)
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Train Loss: {train_loss}')
print(f'Train Accuracy: {train_accuracy}')
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

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

# Save the model
model.save('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Unet_IrisSegmentation_model.h5')

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Flatten the ground truth and predictions for confusion matrix and classification report
y_true = Y_test.flatten()
y_pred = (model.predict(X_test) > 0.5).astype(np.uint8).flatten()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
'''
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
'''

# Classification report
print(classification_report(y_true, y_pred))

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Example visualizations of predictions
import random
n_samples = 5
sample_indices = random.sample(range(len(X_test)), n_samples)
for idx in sample_indices:
    plt.figure(figsize=(10, 3))
    
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[idx])
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(Y_test[idx].squeeze(), cmap='gray')
    plt.title('Ground Truth Mask')
    
    plt.subplot(1, 3, 3)
    pred_mask = model.predict(X_test[idx:idx+1]).squeeze()
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    
    plt.show()
    
    
import pickle
    
with open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/IrisSegUnetModel_WS_26July2024.pkl', 'wb') as f:
    pickle.dump({'X_train': X_train, 'Y_train': Y_train, 'X_val': X_val, 'Y_val': Y_val, 'X_test': X_test, 
                 'Y_test': Y_test, 'model': model, 'history': history}, f)

# Load a sample image and its corresponding segmented mask
def display_image_with_segmented_iris(image, segmented_mask):
    plt.figure(figsize=(10, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Eye Image')
    plt.axis('off')

    # Display segmented iris mask
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_mask, cmap='gray')
    plt.title('Segmented Iris Mask')
    plt.axis('off')

    plt.show()
    
def plot_iris_segment(image, segmented_mask):
    plt.figure(figsize=(10, 5))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Eye Image')
    plt.axis('off')

    # Create an image with only the segmented iris area
    iris_image = np.zeros_like(image)
    iris_image[segmented_mask == 1] = image[segmented_mask == 1]

    # Display the segmented iris
    plt.subplot(1, 2, 2)
    plt.imshow(iris_image)
    plt.title('Segmented Iris')
    plt.axis('off')

    plt.show()
    

# Select a random sample from the test set
import random
sample_index = random.randint(0, len(X_test) - 1)

# Get the original test image and its corresponding mask
sample_image = X_test[sample_index]
sample_image_expanded = np.expand_dims(sample_image, axis=0)  # Expand dimensions for the model
predicted_mask = model.predict(sample_image_expanded)[0].squeeze()
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binarize the mask

# Plot the original image and segmented iris
plot_iris_segment(sample_image, predicted_mask)


'''
# Load the workspace from a file
with open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/IrisSegUnetModel_WS_26July2024.pkl', 'rb') as f:
    workspace = pickle.load(f)
X_train = workspace['X_train']
Y_train = workspace['Y_train']
X_val = workspace['X_val']
Y_val = workspace['Y_val']
X_test = workspace['X_test']
Y_test = workspace['Y_test']
model = workspace['model']
history = workspace['history']
'''
    
'''
# Example index to display (choose any index within the range of your dataset)
index = 100

plt.figure(figsize=(10, 5))

# Display the image
plt.subplot(1, 3, 1)
plt.imshow(images[index])
plt.title('Image')

# Display the corresponding mask
plt.subplot(1, 3, 2)
plt.imshow(masks[index].squeeze(), cmap='gray')
plt.title('Mask')

# Plot the mask on top of the image
plt.subplot(1, 3, 3)
plt.imshow(images[index])
plt.imshow(masks[index], alpha=0.3)  # Overlay mask with transparency
plt.title('Image with Mask Overlay')

plt.show()
'''
'''
# Select an image
img = images[0]  # Ensure this is an RGB image (3 channels)

# Display the original image
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title('Original Image')

# Display individual channels
plt.subplot(1, 4, 2)
plt.imshow(img[:, :, 0], cmap='gray')
plt.title('Red Channel')

plt.subplot(1, 4, 3)
plt.imshow(img[:, :, 1], cmap='gray')
plt.title('Green Channel')

plt.subplot(1, 4, 4)
plt.imshow(img[:, :, 2], cmap='gray')
plt.title('Blue Channel')

plt.tight_layout()
plt.show()
'''