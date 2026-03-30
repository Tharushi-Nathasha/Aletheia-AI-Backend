import cv2
import torch
import base64
import numpy as np
from PIL import Image
from torchvision import transforms

from gradcam import generate_gradcam

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def analyze_video(video_path, model):

    cap = cv2.VideoCapture(video_path)

    frame_scores = []
    heatmap_frames = []
    frame_count = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # SAMPLE EVERY 20 FRAMES (FAST)
        if frame_count % 20 == 0:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            image_resized = image.resize((300, 300))
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                score = torch.sigmoid(output).item()

            frame_scores.append(score)

            #  ALWAYS GENERATE HEATMAP (FIX)
            cam = generate_gradcam(model, image_tensor)

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            original = np.array(image_resized)

            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            _, buffer = cv2.imencode('.jpg', overlay)
            heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

            heatmap_frames.append({
                "frame": frame_count,
                "score": score,
                "heatmap": heatmap_base64
            })

        frame_count += 1

    cap.release()

    if len(frame_scores) == 0:
        return "UNKNOWN", 0, []

    avg_score = sum(frame_scores) / len(frame_scores)
    prediction = "FAKE" if avg_score >= 0.6 else "REAL"

    #  TOP 3 FRAMES ONLY (FAST + CLEAN)
    heatmap_frames = sorted(
        heatmap_frames,
        key=lambda x: x["score"],
        reverse=True
    )[:3]

    return prediction, avg_score, heatmap_frames