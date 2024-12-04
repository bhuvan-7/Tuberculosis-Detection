import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Load your pre-trained model (ensure the path is correct)
model = load_model(r"C:\Users\Bhuvan M\OneDrive\Desktop\TB Detection\tuberculosis_model.h5")

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Convert to RGB if the image is grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to 128x128 (model's expected size)
    image = image.resize((128, 128))
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Normalize the pixel values to the range [0, 1]
    image_array = image_array / 255.0
    
    # Ensure the image has the correct shape (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Print out the shape of the image for debugging
    print(f"Image shape before model: {image_array.shape}")
    
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the file has a valid image extension
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read and preprocess the image
    try:
        image = Image.open(file.stream)
        image_array = preprocess_image(image)
        
        # Ensure the image array has the correct shape (1, 128, 128, 3)
        if image_array.shape != (1, 128, 128, 3):
            return jsonify({'error': f'Image shape mismatch. Expected shape (1, 128, 128, 3), got {image_array.shape}.'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    
    # Make a prediction using the model
    try:
        # Add a try-except block to catch shape errors directly when making predictions
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # for multi-class classification
        print(f"Prediction: {predicted_class}")
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
