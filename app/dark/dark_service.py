import torch
import numpy as np
import cv2
import tempfile
import os

from PIL import Image
from torchvision import transforms

from app.dark.classes import RAGRetinexFormer
from RAG.similarity_search import find_similar_images


MODEL_WEIGHTS_PATH = r'C:\NeuroVisionCombined4\models\retinex_gan_8.pth'

IMAGE_SIZE = 256

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


resize_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


model = RAGRetinexFormer().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_WEIGHTS_PATH,
        map_location=DEVICE
    )
)

model.eval()


def load_reference_tensors(query_image_path, k=3):

    ref_tensors = []

    try:

        refs = find_similar_images(
            query_image_path,
            k=k
        )

        for path, dataset, dist in refs:

            img = Image.open(path).convert("RGB")

            img = resize_transform(img)

            ref_tensors.append(img)

            print(f"Retrieved: {path} | {dataset} | {dist}")

    except Exception as e:

        print(f"Similarity search failed: {e}")

    if len(ref_tensors) == 0:

        dummy_refs = torch.zeros(
            1,
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        return dummy_refs.to(DEVICE)

    ref_tensors = torch.stack(ref_tensors)

    ref_tensors = ref_tensors.unsqueeze(0)

    return ref_tensors.to(DEVICE)


def de_dark(pil_image):

    """
    Low-light enhancement with RAG similarity references.
    """

    temp_path = None

    try:

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        temp_path = temp_file.name

        pil_image.save(temp_path)

        temp_file.close()

        input_tensor = input_transform(
            pil_image
        ).unsqueeze(0).to(DEVICE)

        ref_tensors = load_reference_tensors(
            temp_path,
            k=3
        )

        with torch.no_grad():

            output = model(
                input_tensor,
                ref_tensors
            )

        enhanced = (
            output.squeeze(0)
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        enhanced = (
            enhanced * 0.5 + 0.5
        ).clip(0, 1)

        enhanced_uint8 = (
            enhanced * 255
        ).astype(np.uint8)

        denoised_rgb = cv2.fastNlMeansDenoisingColored(
            enhanced_uint8,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return Image.fromarray(denoised_rgb)

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)