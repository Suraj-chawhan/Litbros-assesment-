from flask import Flask, render_template, request
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import load_model  # Import only model from model.py

# ----------------------
# Device and Model
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("mask_model.pth", device=device)

# ----------------------
# Transform and Prediction
# ----------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
class_names = ["mask","not_mask"]

def predict_image(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0
        confidence = prob*100 if pred==1 else (1-prob)*100
    return class_names[pred], confidence

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__, static_folder="static")

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

# ----------------------
# Run App
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
      
