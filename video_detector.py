import cv2
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
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

        # sample every 10th frame
        if frame_count % 10 == 0:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            image_resized = image.resize((300, 300))
            image_tensor = transform(image).unsqueeze(0)

            # 🔹 Prediction (NO no_grad for Grad-CAM compatibility)
            output = model(image_tensor)
            score = torch.sigmoid(output).item()

            frame_scores.append(score)

            # Grad-CAM (same as image endpoint)
            cam = generate_gradcam(model, image_tensor)

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            original = np.array(image_resized)

            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            # convert to base64 (same as image endpoint)
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

    # take top 5 most suspicious frames
    heatmap_frames = sorted(
        heatmap_frames,
        key=lambda x: x["score"],
        reverse=True
    )[:5]

    return prediction, avg_score, heatmap_frames