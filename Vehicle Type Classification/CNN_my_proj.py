import os
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size):
    image = load_img(image, target_size=target_size) 
    image = img_to_array(image)  
    image = image / 255.0  
    image = tf.expand_dims(image, axis=0)  
    return image

app = Flask(__name__)
model = load_trained_model('cnn_model_final .h5') 
class_labels = {
    0: 'Auto Rickshaws', 1: 'Bicycle', 2: 'Bus', 3: 'Cars', 4: 'Jeep',
    5: 'Motorcycles', 6: 'Planes', 7: 'Ships', 8: 'Trains', 9: 'Truck'
} 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    image_file = request.files['image']

    try:
       
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, image_file.filename)
            image_file.save(image_path)

          
            image = preprocess_image(image_path, target_size=(150, 150))  

         
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













