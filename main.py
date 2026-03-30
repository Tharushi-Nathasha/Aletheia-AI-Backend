from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import shutil
import cv2
import numpy as np
import base64
import os

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

from video_detector import analyze_video
from gradcam import generate_gradcam

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL ----------------
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

model = AletheiaModel()
model.load_state_dict(torch.load("models/celebdf_final_model.pth", map_location="cpu"))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- FACE DETECTION ----------------
def detect_face(image_np):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0

# ---------------- IMAGE API ----------------
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    image_np = np.array(image)

    # FACE VALIDATION
    if not detect_face(image_np):
        raise HTTPException(
            status_code=400,
            detail="No human face detected. Please upload a clear image containing a face."
        )

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

# ---------------- VIDEO API ----------------
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    try:
        # SAVE TEMP VIDEO
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PROCESS VIDEO
        prediction, score, heatmaps = analyze_video(temp_path, model)

        return {
            "prediction": prediction,
            "confidence": float(score),
            "frames": heatmaps
        }

    finally:
        # AUTO DELETE FILE
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------- AUTH ----------------
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

# ---------------- SIGNUP ----------------
@app.post("/signup")
def signup(user: User):

    if user.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    users_db[user.username] = hash_password(user.password)

    return {"message": "User created successfully"}

# ---------------- LOGIN ----------------
@app.post("/login")
def login(user: User):

    if user.username not in users_db:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(user.password, users_db[user.username]):
        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_token({"sub": user.username})

    return {"access_token": token}