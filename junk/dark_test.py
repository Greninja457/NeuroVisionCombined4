import matplotlib.pyplot as plt
from PIL import Image
from app.dark.dark_service import de_dark

# Define the path to a test image
test_image_path = r"C:\NeuroVisionCombined4\data\LOL-v2\Real_captured\Test\Low\low00692.png"

# 1. Load the original image
original_pil = Image.open(test_image_path).convert('RGB')

# 2. Apply the enhancement function
enhanced_pil = de_dark(original_pil)

# Save the enhanced image without showing a plot
enhanced_pil.save('output.png')





# import math
# import numpy as np
# import torch

# from PIL import Image
# from torchvision import transforms

# from app.dark.classes import RAGRetinexFormer
# from RAG.similarity_search import find_similar_images


# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# MODEL_PATH = r"C:\NeuroVisionCombined4\models\retinex_gan_8.pth"

# INPUT_IMAGE = r"C:\NeuroVisionCombined4\data\LOL-v2\Real_captured\Test\Low\low00690.png"

# # INPUT_IMAGE = r"C:\NeuroVisionCombined4\data\LOL-v2\Real_captured\Test\Low\low00698.png"

# OUTPUT_IMAGE = r"C:\NeuroVisionCombined4\app\dark\output.png"

# PATCH_SIZE = 256

# STRIDE = 64


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5] * 3, [0.5] * 3)
# ])


# def tensor_to_numpy(tensor):

#     tensor = tensor.squeeze(0)

#     tensor = (tensor + 1) / 2

#     tensor = tensor.clamp(0, 1)

#     tensor = tensor.permute(1, 2, 0).cpu().numpy()

#     return tensor


# def numpy_to_pil(array):

#     array = np.clip(array * 255, 0, 255).astype("uint8")

#     return Image.fromarray(array)


# def load_reference_tensors(image_path, k=5):

#     refs = find_similar_images(
#         image_path,
#         k=k
#     )

#     ref_tensors = []

#     resize_transform = transforms.Compose([
#         transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5] * 3, [0.5] * 3)
#     ])

#     for path, dataset, dist in refs:

#         img = Image.open(path).convert("RGB")

#         img = resize_transform(img)

#         ref_tensors.append(img)

#         print(f"Retrieved: {path} | {dataset} | {dist}")

#     ref_tensors = torch.stack(ref_tensors)

#     ref_tensors = ref_tensors.unsqueeze(0)

#     return ref_tensors.to(DEVICE)


# def create_weight_mask(size):

#     y = np.hanning(size)

#     x = np.hanning(size)

#     mask = np.outer(y, x)

#     mask = mask / mask.max()

#     return mask.astype(np.float32)


# def split_image(image):

#     width, height = image.size

#     patches = []

#     positions = []

#     for top in range(0, height, STRIDE):

#         for left in range(0, width, STRIDE):

#             bottom = min(top + PATCH_SIZE, height)

#             right = min(left + PATCH_SIZE, width)

#             patch = image.crop((left, top, right, bottom))

#             patch_np = np.array(patch)

#             pad_h = PATCH_SIZE - patch_np.shape[0]

#             pad_w = PATCH_SIZE - patch_np.shape[1]

#             if pad_h > 0 or pad_w > 0:

#                 patch_np = np.pad(
#                     patch_np,
#                     (
#                         (0, pad_h),
#                         (0, pad_w),
#                         (0, 0)
#                     ),
#                     mode="reflect"
#                 )

#             patch = Image.fromarray(patch_np)

#             patches.append(patch)

#             positions.append((left, top))

#     return patches, positions


# def main():

#     model = RAGRetinexFormer().to(DEVICE)

#     model.load_state_dict(
#         torch.load(MODEL_PATH, map_location=DEVICE)
#     )

#     model.eval()

#     image = Image.open(INPUT_IMAGE).convert("RGB")

#     width, height = image.size

#     patches, positions = split_image(image)

#     ref_tensors = load_reference_tensors(
#         INPUT_IMAGE,
#         k=5
#     )

#     output_accumulator = np.zeros(
#         (height, width, 3),
#         dtype=np.float32
#     )

#     weight_accumulator = np.zeros(
#         (height, width, 1),
#         dtype=np.float32
#     )

#     weight_mask = create_weight_mask(PATCH_SIZE)

#     weight_mask = weight_mask[..., np.newaxis]

#     total = len(patches)

#     for idx, (patch, (left, top)) in enumerate(zip(patches, positions)):

#         input_tensor = transform(patch)

#         input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

#         with torch.no_grad():

#             output = model(
#                 input_tensor,
#                 ref_tensors
#             )

#         output_np = tensor_to_numpy(output)

#         valid_h = min(PATCH_SIZE, height - top)

#         valid_w = min(PATCH_SIZE, width - left)

#         output_np = output_np[:valid_h, :valid_w]

#         current_mask = weight_mask[:valid_h, :valid_w]

#         output_accumulator[
#             top:top + valid_h,
#             left:left + valid_w
#         ] += output_np * current_mask

#         weight_accumulator[
#             top:top + valid_h,
#             left:left + valid_w
#         ] += current_mask

#         print(f"Processed patch {idx + 1}/{total}")

#     final_output = output_accumulator / (
#         weight_accumulator + 1e-8
#     )

#     final_image = numpy_to_pil(final_output)

#     final_image.save(OUTPUT_IMAGE)

#     print(f"Saved enhanced image to: {OUTPUT_IMAGE}")


# if __name__ == "__main__":
#     main()
