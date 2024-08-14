# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:14:43 2024

@author: Vinayak
"""
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the pre-trained U-Net model
model = load_model('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Unet_IrisSegmentation_model.h5')

# Global variable to store the segmented iris image for saving
global_iris_image = None

def segment_iris(image):
    # Resize image
    resized_image = cv2.resize(image, (224, 224))
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    resized_image_expanded = np.expand_dims(resized_image_rgb, axis=0) / 255.0

    # Predict the mask
    predicted_mask = model.predict(resized_image_expanded)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8).squeeze()

    # Create an image with only the segmented iris area
    iris_image = np.ones_like(resized_image_rgb) * 255  # Start with a white background
    iris_image[predicted_mask == 1] = resized_image_rgb[predicted_mask == 1]

    return resized_image_rgb, iris_image

def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            # Open and process the image
            image = cv2.imread(file_path)
            original_image, iris_image = segment_iris(image)

            global global_iris_image
            global_iris_image = iris_image  # Store the segmented iris image

            # Convert images to PIL format for Tkinter
            original_image_pil = Image.fromarray(original_image)
            iris_image_pil = Image.fromarray(iris_image)

            # Convert PIL images to ImageTk format
            original_image_tk = ImageTk.PhotoImage(original_image_pil)
            iris_image_tk = ImageTk.PhotoImage(iris_image_pil)

            # Update labels with images
            original_image_label.config(image=original_image_tk)
            original_image_label.image = original_image_tk

            iris_image_label.config(image=iris_image_tk)
            iris_image_label.image = iris_image_tk

            # Enable the save button
            save_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", str(e))

def save_image():
    if global_iris_image is not None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            try:
                # Convert the segmented iris image to PIL format
                iris_image_pil = Image.fromarray(global_iris_image)
                # Save the image
                iris_image_pil.save(file_path)
                messagebox.showinfo("Saved", f"Image saved as {file_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    else:
        messagebox.showwarning("No Image", "No segmented iris image to save")

# Create main window
root = tk.Tk()
root.title("Iris Segmentation App")

# Set fixed window size
root.geometry("800x400")  # Adjust the size as needed

# Create and pack widgets
upload_button = tk.Button(root, text="Upload Eye Image", command=open_file)
upload_button.pack(pady=10)

save_button = tk.Button(root, text="Save Iris Image", command=save_image, state=tk.DISABLED)
save_button.pack(pady=10)

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

# Create and pack titles and image labels
original_image_title = tk.Label(frame, text="Original Eye Image")
original_image_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="n")

original_image_label = tk.Label(frame)
original_image_label.grid(row=1, column=0, padx=10, pady=10)

iris_image_title = tk.Label(frame, text="Segmented Iris Image")
iris_image_title.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="n")

iris_image_label = tk.Label(frame)
iris_image_label.grid(row=1, column=1, padx=10, pady=10)

# Run the application
root.mainloop()
