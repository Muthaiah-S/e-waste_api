from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend access

# Load CNN Model
model = load_model("e_waste_model.h5")

# Define e-waste categories
class_labels = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
                'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file.save("uploaded_image.jpg")

    print("✅ File received:", file.filename)
    print("📷 Content-Type:", file.content_type)

    # Preprocess image
    img = image.load_img("uploaded_image.jpg", target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict category
    predictions = model.predict(img_array)
    predicted_category = class_labels[np.argmax(predictions)]

    # Estimate price
    price_estimate = np.random.randint(100, 5000)

    return jsonify({
        "category": predicted_category,
        "price": price_estimate,
        "reuse_probability": np.random.randint(10, 90)
    })

if __name__ == "__main__":
    app.run(debug=True)
