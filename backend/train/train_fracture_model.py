import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import ChestMNIST
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# ==== Hyperparameters ====
BATCH_SIZE = 32
EPOCHS = 6
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "backend/models/fracture_detector.pth"

# ==== Transformations (with data augmentation) ====
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==== Load Dataset ====
train_dataset = ChestMNIST(split="train", download=True, size=224, transform=train_transform)
test_dataset = ChestMNIST(split="test", download=True, size=224, transform=test_transform)

# ==== Compute pos_weight for BCEWithLogitsLoss ====
label_counts = np.sum(train_dataset.labels, axis=0)
pos_weights = (len(train_dataset) - label_counts) / (label_counts + 1e-5)
pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

# ==== Data Loaders ====
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Model Setup ====
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 14)  # 14 diseases (multi-label)

# Load existing weights if available
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("No pretrained model found. Starting from scratch.")

model.to(DEVICE)

# ==== Loss, Optimizer, Scheduler ====
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==== Training Loop ====
best_loss = float('inf')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.sigmoid(outputs)

        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    scheduler.step()

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    auc = roc_auc_score(all_labels_np, all_preds_np, average="macro")
    f1 = f1_score(all_labels_np, (all_preds_np > 0.5).astype(float), average="macro")

    print(f"\nEpoch {epoch+1}: Avg Loss = {avg_loss:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")

    # Print per-label AUC and F1
    for i, label in enumerate(train_dataset.info["label"].values()):
        class_auc = roc_auc_score(all_labels_np[:, i], all_preds_np[:, i])
        class_f1 = f1_score(all_labels_np[:, i], (all_preds_np[:, i] > 0.5).astype(float))
        print(f"🔍 {label:20s} | AUC: {class_auc:.3f} | F1: {class_f1:.3f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f" Saved new best model (loss: {avg_loss:.4f})")

print(f"Final model saved to {MODEL_PATH}")