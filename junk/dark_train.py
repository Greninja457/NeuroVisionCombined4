import os
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from app.dark.classes import (
    RAGRetinexFormer,
    PatchDiscriminator
)

from RAG.similarity_search import find_similar_images


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LOW_DIR = r"C:\\NeuroVisionCombined4\\data\\LOL-v2\\Real_captured\\Train\\Low"

NORMAL_DIR = r"C:\\NeuroVisionCombined4\\data\\LOL-v2\\Real_captured\\Train\\Normal"

SAVE_PATH = r"app\\dark\\low_light.pth"

BATCH_SIZE = 2

EPOCHS = 1

LR = 1e-4

IMAGE_SIZE = 256


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


class LOLPairDataset(Dataset):

    def __init__(self, low_dir, normal_dir):

        self.low_dir = low_dir

        self.normal_dir = normal_dir

        self.low_files = sorted(os.listdir(low_dir))

    def __len__(self):

        return len(self.low_files)

    def __getitem__(self, idx):

        low_name = self.low_files[idx]

        normal_name = low_name.replace(
            "low",
            "normal"
        )

        low_path = os.path.join(
            self.low_dir,
            low_name
        )

        normal_path = os.path.join(
            self.normal_dir,
            normal_name
        )

        low_img = Image.open(low_path).convert("RGB")

        normal_img = Image.open(normal_path).convert("RGB")

        low_tensor = transform(low_img)

        normal_tensor = transform(normal_img)

        refs = find_similar_images(
            low_path,
            k=3
        )

        ref_tensors = []

        for path, dataset, dist in refs:

            ref_img = Image.open(path).convert("RGB")

            ref_tensors.append(
                transform(ref_img)
            )

        ref_tensors = torch.stack(ref_tensors)

        return (
            low_tensor,
            normal_tensor,
            ref_tensors
        )


dataset = LOLPairDataset(
    LOW_DIR,
    NORMAL_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


generator = RAGRetinexFormer().to(DEVICE)

discriminator = PatchDiscriminator().to(DEVICE)


l1_loss = nn.L1Loss()

bce_loss = nn.BCEWithLogitsLoss()


g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=LR
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=LR
)


for epoch in range(EPOCHS):

    loop = tqdm(loader)

    for low_img, gt_img, ref_imgs in loop:

        low_img = low_img.to(DEVICE)

        gt_img = gt_img.to(DEVICE)

        ref_imgs = ref_imgs.to(DEVICE)

        # ---------------------
        # Generator
        # ---------------------

        fake_img = generator(
            low_img,
            ref_imgs
        )

        pred_fake = discriminator(fake_img)

        adv_loss = bce_loss(
            pred_fake,
            torch.ones_like(pred_fake)
        )

        pixel_loss = l1_loss(
            fake_img,
            gt_img
        )

        g_loss = pixel_loss + 0.01 * adv_loss

        g_optimizer.zero_grad()

        g_loss.backward()

        g_optimizer.step()

        # ---------------------
        # Discriminator
        # ---------------------

        pred_real = discriminator(gt_img)

        pred_fake = discriminator(
            fake_img.detach()
        )

        real_loss = bce_loss(
            pred_real,
            torch.ones_like(pred_real)
        )

        fake_loss = bce_loss(
            pred_fake,
            torch.zeros_like(pred_fake)
        )

        d_loss = (
            real_loss + fake_loss
        ) / 2

        d_optimizer.zero_grad()

        d_loss.backward()

        d_optimizer.step()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

        loop.set_postfix(
            g_loss=g_loss.item(),
            d_loss=d_loss.item()
        )

    torch.save(
        generator.state_dict(),
        SAVE_PATH
    )

    print(f"Saved model to {SAVE_PATH}")