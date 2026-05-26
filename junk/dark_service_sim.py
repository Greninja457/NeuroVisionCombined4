import torch
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
from app.dark.classes import RAGRetinexFormer

MODEL_WEIGHTS_PATH = r'C:\NeuroVisionCombined4\models\retinex_gan_8.pth'
IMAGE_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_transform = transforms.Compose([
    # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

model = RAGRetinexFormer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.eval()

def de_dark(pil_image):
    """
    Standalone dark image enhancement function.
    Logic: Encapsulates Init -> Load -> Inference -> Post-processing.
    """
    
    input_tensor = input_transform(pil_image).unsqueeze(0).to(DEVICE)
    
    dummy_refs = torch.zeros(1, 1, 3, IMAGE_SIZE//4, IMAGE_SIZE//4).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor, dummy_refs)

    enhanced = (output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    enhanced_uint8 = (enhanced * 255).astype(np.uint8)

    denoised_rgb = cv2.fastNlMeansDenoisingColored(enhanced_uint8, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

    return Image.fromarray(denoised_rgb)

