import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Unet_IrisSegmentation_model.tflite')
interpreter.allocate_tensors()

# Get model input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class IrisSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Segmentation App version 1.0.0")
        self.root.geometry("800x600")  # Initial size; adjust as needed

        # Global variable to store the segmented iris image for saving
        self.global_iris_image = None

        # Create and pack widgets
        self.upload_button = tk.Button(self.root, text="Upload Eye Image", command=self.open_file)
        self.upload_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.save_button = tk.Button(self.root, text="Save Iris Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Create frames for content and logo
        self.content_frame = tk.Frame(self.root)
        self.content_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.logo_frame = tk.Frame(self.root)
        self.logo_frame.grid(row=2, column=1, padx=10, pady=10, sticky="ne")

        # Create and place widgets in the content frame
        self.original_image_title = tk.Label(self.content_frame, text="Eye")
        self.original_image_title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="n")

        self.original_image_label = tk.Label(self.content_frame)
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.iris_image_title = tk.Label(self.content_frame, text="Iris")
        self.iris_image_title.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="n")

        self.iris_image_label = tk.Label(self.content_frame)
        self.iris_image_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Add content to the logo frame
        self.powered_by_label = tk.Label(self.logo_frame, text="Powered by", font=("Arial", 10))
        self.powered_by_label.pack(side=tk.LEFT, padx=(0, 5))

        # Initialize logo image after creating the Tkinter window
        self.load_logo_image()

        # Configure row and column weights for resizing
        self.root.grid_rowconfigure(1, weight=1)  # Content frame row
        self.root.grid_columnconfigure(0, weight=1)  # Left column
        self.root.grid_columnconfigure(1, weight=1)  # Right column
        self.content_frame.grid_rowconfigure(1, weight=1)  # Image rows
        self.content_frame.grid_columnconfigure(0, weight=1)  # Original image column
        self.content_frame.grid_columnconfigure(1, weight=1)  # Iris image column

    def load_logo_image(self):
        try:
            logo_image = Image.open('C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/TeleMedClogo.png')  # Replace with your logo path
            logo_image = logo_image.resize((150, 40))  # Resize if needed
            self.global_logo_image_tk = ImageTk.PhotoImage(logo_image)
            self.logo_label = tk.Label(self.logo_frame, image=self.global_logo_image_tk)
            self.logo_label.pack(side=tk.LEFT)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load logo image: {e}")

    def segment_iris(self, image):
        # Resize image to the input size expected by the model
        input_size = (224, 224)
        resized_image = cv2.resize(image, input_size)
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image_expanded = np.expand_dims(resized_image_rgb, axis=0) / 255.0
        resized_image_expanded = resized_image_expanded.astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], resized_image_expanded)
        interpreter.invoke()

        # Get the output tensor
        predicted_mask = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8).squeeze()

        # Apply Gaussian blur to smooth the edges
        smooth_mask = cv2.GaussianBlur(predicted_mask, (5, 5), 0)

        # Resize the mask to the original image size
        original_height, original_width = image.shape[:2]
        smooth_mask_resized_to_original = cv2.resize(smooth_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        # Create an image with only the segmented iris area using the original image
        iris_image = np.ones_like(image) * 255  # Start with a white background
        mask_threshold = smooth_mask_resized_to_original > 0.5  # Convert to binary mask after smoothing
        iris_image[mask_threshold] = image[mask_threshold]

        # Create an image with only the segmented iris area for display (resized)
        iris_image_display = np.ones_like(resized_image_rgb) * 255
        display_mask_threshold = smooth_mask > 0.5
        iris_image_display[display_mask_threshold] = resized_image_rgb[display_mask_threshold]

        return resized_image_rgb, iris_image_display, iris_image

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                # Open and process the image
                image = cv2.imread(file_path)
                original_image, iris_image_display, iris_image_original = self.segment_iris(image)

                self.global_iris_image = iris_image_original  # Store the original segmented iris image

                # Convert images to PIL format for Tkinter
                original_image_pil = Image.fromarray(original_image)
                iris_image_display_pil = Image.fromarray(iris_image_display)

                # Convert PIL images to ImageTk format
                original_image_tk = ImageTk.PhotoImage(original_image_pil)
                iris_image_tk = ImageTk.PhotoImage(iris_image_display_pil)

                # Update labels with images
                self.original_image_label.config(image=original_image_tk)
                self.original_image_label.image = original_image_tk

                self.iris_image_label.config(image=iris_image_tk)
                self.iris_image_label.image = iris_image_tk

                # Enable the save button
                self.save_button.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Error", str(e))

    def save_image(self):
        if self.global_iris_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    # Convert the segmented iris image to PIL format and save in RGB
                    iris_image_rgb = cv2.cvtColor(self.global_iris_image, cv2.COLOR_BGR2RGB)
                    iris_image_pil = Image.fromarray(iris_image_rgb)
                    iris_image_pil.save(file_path)
                    messagebox.showinfo("Saved", f"Image saved as {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))
        else:
            messagebox.showwarning("No Image", "No segmented iris image to save")

# Create the main window and start the application
root = tk.Tk()
app = IrisSegmentationApp(root)
root.mainloop()