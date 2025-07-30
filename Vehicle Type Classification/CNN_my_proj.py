import os
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify

# Load the trained model
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

# Preprocess the input image
def preprocess_image(image, target_size):
    image = load_img(image, target_size=target_size)  # Load image
    image = img_to_array(image)  # Convert to array
    image = image / 255.0  # Normalize
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Flask App
app = Flask(__name__)
model = load_trained_model('cnn_model_final .h5')  # Replace with your model's path
class_labels = {
    0: 'Auto Rickshaws', 1: 'Bicycle', 2: 'Bus', 3: 'Cars', 4: 'Jeep',
    5: 'Motorcycles', 6: 'Planes', 7: 'Ships', 8: 'Trains', 9: 'Truck'
}  # Update based on your dataset

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    image_file = request.files['image']

    try:
        # Save the uploaded image temporarily in a writable temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, image_file.filename)
            image_file.save(image_path)

            # Preprocess the image
            image = preprocess_image(image_path, target_size=(150, 150))  # Use your input size

            # Predict using the model
            predictions = model.predict(image)
            predicted_class = class_labels[predictions.argmax()]
            confidence = float(predictions.max())

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)













