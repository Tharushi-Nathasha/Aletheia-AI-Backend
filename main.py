from fastapi import FastAPI, UploadFile, File, HTTPException
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
import os

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

# ================== APP ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== ROOT ==================
@app.get("/")
def home():
    return {"message": "Aletheia backend is running 🚀"}

# ================== MODEL ==================
class AletheiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        self.backbone.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

model = None

def get_model():
    global model
    if model is None:
        try:
            model_path = "models/celebdf_final_model.pth"

            if not os.path.exists(model_path):
                raise Exception("Model file not found")

            model = AletheiaModel()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    return model

# ================== TRANSFORM ==================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================== IMAGE API ==================
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    try:
        model = get_model()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_resized = image.resize((300, 300))
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            score = torch.sigmoid(output).item()

        prediction = "FAKE" if score >= 0.6 else "REAL"

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== VIDEO API ==================
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    try:
        model = get_model()

        temp_path = f"temp_{file.filename}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction, score, heatmaps = analyze_video(temp_path, model)

        return {
            "prediction": prediction,
            "confidence": float(score),
            "frames": heatmaps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================== AUTH ==================
SECRET_KEY = "secret123"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users_db = {}

class User(BaseModel):
    username: str
    password: str

def hash_password(password):
    return pwd_context.hash(password[:72])

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=2)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/signup")
def signup(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    users_db[user.username] = hash_password(user.password)

    return {"message": "User created successfully"}

@app.post("/login")
def login(user: User):
    if user.username not in users_db:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(user.password, users_db[user.username]):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_token({"sub": user.username})

    return {"access_token": token}