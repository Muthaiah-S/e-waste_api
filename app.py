from flask import Flask, request, jsonify, make_response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for all routes and all origins (for testing/dev)
# For production, replace "*" with your frontend domain
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load your trained CNN model
model = load_model("e_waste_model.h5")

# Class labels for prediction
class_labels = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
                'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return make_response(jsonify({"error": "No file uploaded"}), 400)

    file = request.files["file"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    print("✅ File received:", file.filename)
    print("📷 Content-Type:", file.content_type)

    try:
        # Preprocess the image
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_category = class_labels[np.argmax(predictions)]

        # Generate dummy price and reuse probability
        price_estimate = np.random.randint(100, 5000)
        reuse_chance = np.random.randint(10, 90)

        # Send response
        response = jsonify({
            "category": predicted_category,
            "price": price_estimate,
            "reuse_probability": reuse_chance
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)

if __name__ == "__main__":
    app.run(debug=True)
