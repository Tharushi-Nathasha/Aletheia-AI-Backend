import torch
import cv2
import numpy as np

def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # hook last conv layer
    target_layer = model.backbone.features[-1]

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output.squeeze()

    model.zero_grad()
    score.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(cam, (300, 300))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam