from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from video_detector import analyze_video
import shutil
from gradcam import generate_gradcam
import base64

import cv2
import numpy as np

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  MODEL ARCHITECTURE 
class AletheiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        self.backbone.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1536,512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,1)
        )

    def forward(self,x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

#  LOAD MODEL 
model = AletheiaModel()
model.load_state_dict(torch.load("models/celebdf_final_model.pth", map_location="cpu"))
model.eval()

# IMAGE TRANSFORM 
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# API ENDPOINT 
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_resized = image.resize((300, 300))

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        score = torch.sigmoid(output).item()

    prediction = "FAKE" if score >= 0.6 else "REAL"

    # Grad-CAM
    cam = generate_gradcam(model, image_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = np.array(image_resized)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', overlay)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "prediction": prediction,
        "confidence": float(score),
        "heatmap": heatmap_base64
    }

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, score, heatmaps = analyze_video(temp_path, model)

    return {
        "prediction": prediction,
        "confidence": float(score),
        "frames": heatmaps  # SAME STYLE AS IMAGE HEATMAP
    }