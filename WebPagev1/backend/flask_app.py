from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import os
from flask import send_file

app = Flask(__name__)
CORS(app)

# Load the TensorFlow Lite model
# model_path = 'C:/Users/Vinayak/TeleMedC/IrisSegmentationProject/Unet_IrisSegmentation_model.tflite'
model_path = '/home/TeleMedC/mysite/Unet_IrisSegmentation_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('frontend', path)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Load and preprocess the image
        original_image = Image.open(io.BytesIO(file.read()))

        image = preprocess_image(original_image)

        # Perform segmentation
        output_image = process_segmentation(image)

        # Postprocess the output
        segmented_mask = postprocess_output(output_image)

        # Create an image with only the segmented iris area at original resolution
        original_resolution_iris_image = create_iris_image(original_image, segmented_mask)

        # Save the original resolution segmented image in the static directory
        save_path = os.path.join('static', 'segmented_image.png')
        original_resolution_iris_image.save(save_path)

        # Resize images for display (if needed)
        original_resized_image = resize_image(original_image, (400, 400))
        iris_resized_image = resize_image(original_resolution_iris_image, (400, 400))

        # Convert images to bytes for displaying
        original_img_bytes = io.BytesIO()
        original_resized_image.save(original_img_bytes, format='PNG')
        original_img_bytes.seek(0)

        iris_img_bytes = io.BytesIO()
        iris_resized_image.save(iris_img_bytes, format='PNG')
        iris_img_bytes.seek(0)

        return jsonify({
            'original_image': 'data:image/png;base64,' + base64.b64encode(original_img_bytes.getvalue()).decode(),
            'segmented_image': 'data:image/png;base64,' + base64.b64encode(iris_img_bytes.getvalue()).decode(),
            'original_resolution_segmented_image': url_for('static', filename='segmented_image.png')
        })

    return jsonify({'error': 'File not uploaded'}), 400

def preprocess_image(image):
    input_shape = interpreter.get_input_details()[0]['shape']
    image = image.resize((input_shape[2], input_shape[1]))  # Resize to (height, width)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Normalize if required
    if len(input_shape) == 4:
        image = np.expand_dims(image, axis=0)  # Add batch dimension if needed
    return image

def process_segmentation(image):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])
    return output

def postprocess_output(output_image):
    if output_image.ndim == 4:
        output_image = output_image[0]  # Remove batch dimension if present
    if output_image.shape[-1] == 1:
        output_image = output_image.squeeze(axis=-1)  # Remove single channel dimension
    output_image = (output_image * 255).astype(np.uint8)  # Scale to [0, 255]
    return Image.fromarray(output_image)

def create_iris_image(original_image, mask_image):
    # Ensure the mask is in the same size as the original image
    mask_image = mask_image.resize(original_image.size)
    mask_image = mask_image.convert("L")  # Ensure mask is in grayscale

    # Create a new image with a white background
    iris_image = Image.new("RGBA", original_image.size, (255, 255, 255, 255))

    # Apply the mask to the original image
    original_image = original_image.convert("RGBA")
    iris_image.paste(original_image, mask=mask_image)

    return iris_image

def resize_image(image, max_size):
    # Ensure max_size is an integer
    if isinstance(max_size, tuple):
        max_size = max(max_size)  # Use the larger of the two dimensions for resizing

    # Get the original size
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new dimensions
    if original_width > original_height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the desired size and paste the resized image onto it
    background = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    paste_x = (max_size - new_width) // 2
    paste_y = (max_size - new_height) // 2
    background.paste(resized_image, (paste_x, paste_y))

    return background



# if __name__ == '__main__':
#     app.run(debug=True,port=8000)
