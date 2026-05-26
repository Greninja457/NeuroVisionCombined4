import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("C:/NeuroVisionCombined4")

SAVE_PATH = (
    BASE_DIR / "models" / "pth" / "classifier.pth"
)

TARGET_LABELS = [
    "foggy",
    "blurry",
    "dark",
    "damaged",
    "normal"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# DEGRADED DATASETS
# ============================================================

DATASETS = {

    "damaged": [
        BASE_DIR / "data" / "damaged_media" / "images"
    ],

    "blurry": [
        BASE_DIR / "data" / "gopro_deblur" / "blur" / "images"
    ],

    "dark": [
        BASE_DIR / "data" / "LOL-v2" / "Real_captured" / "Train" / "Low",

        BASE_DIR / "data" / "LOL-v2" / "Synthetic" / "Train" / "Low"
    ],

    "foggy": [
        BASE_DIR / "data" / "RESIDE-6K" / "train" / "hazy",

        BASE_DIR / "data" / "RESIDE-ITS" / "hazy"
    ]
}

# ============================================================
# NORMAL / CLEAN DATASETS
# ============================================================

NORMAL_SOURCES = [

    BASE_DIR / "data" / "gopro_deblur" / "sharp" / "images",

    BASE_DIR / "data" / "LOL-v2" / "Real_captured" / "Train" / "Normal",

    BASE_DIR / "data" / "LOL-v2" / "Synthetic" / "Train" / "Normal",

    BASE_DIR / "data" / "RESIDE-6K" / "train" / "GT",

    BASE_DIR / "data" / "RESIDE-ITS" / "clear",

    BASE_DIR / "data" / "coco_minitrain_10k" / "images" / "train2017"
]

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# HELPERS
# ============================================================

def collect_images(folder):

    images = []

    if not folder.exists():
        print(f"Folder Missing: {folder}")
        return images

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))

    return images

# ============================================================
# BUILD DATASET
# ============================================================

all_samples = []

# ------------------------------------------------------------
# DEGRADED IMAGES
# ------------------------------------------------------------

FOGGY_LIMIT = 6000

for label, folders in DATASETS.items():

    total = 0

    for folder in folders:

        images = collect_images(folder)

        # ----------------------------------------------------
        # Cap foggy dataset because it is extremely large
        # ----------------------------------------------------

        if label == "foggy" and len(images) > FOGGY_LIMIT:

            images = random.sample(
                images,
                FOGGY_LIMIT
            )

        total += len(images)

        for img in images:
            all_samples.append((str(img), label))

    print(f"{label} images collected: {total}")

# ------------------------------------------------------------
# NORMAL IMAGES
# ------------------------------------------------------------

normal_images = []

for folder in NORMAL_SOURCES:

    images = collect_images(folder)

    print(f"Normal images collected from {folder}: {len(images)}")

    normal_images.extend(images)

# ------------------------------------------------------------
# IMPORTANT BALANCING
# ------------------------------------------------------------

# Foggy dataset is massive (~20k images)
# COCO adds another 10k
# So we cap normal images.

NORMAL_LIMIT = 8000

if len(normal_images) > NORMAL_LIMIT:

    normal_images = random.sample(
        normal_images,
        NORMAL_LIMIT
    )

print(f"\nUsing {len(normal_images)} normal images")

for img in normal_images:
    all_samples.append((str(img), "normal"))

# ------------------------------------------------------------
# SHUFFLE
# ------------------------------------------------------------

random.shuffle(all_samples)

# ============================================================
# LABEL ENCODING
# ============================================================

label_to_idx = {
    label: idx
    for idx, label in enumerate(TARGET_LABELS)
}

# ============================================================
# DATASET CLASS
# ============================================================

class NeuroVisionDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.zeros(len(TARGET_LABELS))

        label_tensor[label_to_idx[label]] = 1.0

        return image, label_tensor

# ============================================================
# DATASET
# ============================================================

dataset = NeuroVisionDataset(
    all_samples,
    transform=train_transform
)

# ============================================================
# CLASS BALANCING
# ============================================================

labels = [label for _, label in all_samples]

class_counts = Counter(labels)

print("\n================================================")
print("CLASS DISTRIBUTION")
print("================================================")

for k, v in class_counts.items():
    print(f"{k}: {v}")

sample_weights = []

for _, label in all_samples:

    sample_weights.append(
        1.0 / class_counts[label]
    )

sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)

# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0,

    pin_memory=True
)

# ============================================================
# MODEL
# ============================================================

class NeuroVisionModel(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_ftrs,
            num_classes
        )

    def forward(self, x):
        return self.model(x)

# ============================================================
# LOAD MODEL
# ============================================================

model = NeuroVisionModel(
    num_classes=len(TARGET_LABELS)
).to(DEVICE)

# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LR
)

# ============================================================
# TRAINING
# ============================================================

print("\n================================================")
print("TRAINING STARTED")
print("================================================\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    loop = tqdm(loader)

    for images, labels in loop:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    avg_loss = running_loss / len(loader)

    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}\n")

# ============================================================
# SAVE MODEL
# ============================================================

SAVE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    SAVE_PATH
)

print("\n================================================")
print("MODEL SAVED")
print("================================================")
print(SAVE_PATH)

if __name__ == "__main__":
    pass