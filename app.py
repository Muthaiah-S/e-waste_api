from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("e_waste_model_resaved.keras", compile=False)

categories = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

def predict_e_waste(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    prediction = model.predict(img_tensor)
    predicted_index = np.argmax(prediction)
    predicted_class = categories[predicted_index]
    confidence = round(float(np.max(prediction)) * 100, 2)
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            predicted_class, confidence = predict_e_waste(filepath)
            return render_template("index.html", prediction=predicted_class, confidence=confidence, img_path=filepath)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
