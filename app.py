from flask import Flask, render_template, request
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = load_model("mask_model.pth", device=device)

# Flask app
app = Flask(__name__, static_folder="static")

# Image preprocessing (must match training pipeline!)
transform = transforms.Compose([
    transforms.Resize((128, 128)),   
    transforms.ToTensor(),
])


# predict  image 
def predict_image(model, image_path, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "Mask" if prob >= 0.5 else "No Mask"
    confidence = prob if prediction == "Mask" else 1 - prob
    return prediction, round(confidence * 100, 2)

# home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, uploaded_img = None, None, None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            uploaded_img = os.path.join("static", file.filename)
            file.save(uploaded_img)

            prediction, confidence = predict_image(model, uploaded_img, device=device)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           uploaded_img=uploaded_img)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
