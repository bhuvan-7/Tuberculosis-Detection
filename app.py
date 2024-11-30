import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Configuration for uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = 'parkinsons_vgg16_model.h5'  # Replace with the path to your model
model = load_model(MODEL_PATH)

# Define image dimensions based on training
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Ensure this matches your model's training input size

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(file_path):
    try:
        # Resize image to the size expected by the model
        img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Resize to 128x128
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)

        # Interpret the prediction
        return "Parkinson" if prediction[0][0] > 0.5 else "Healthy"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform prediction
            result = predict_image(file_path)

            # Render result back to user
            return render_template('index.html', result=result, image_path=file_path)

    return render_template('index.html')

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)
