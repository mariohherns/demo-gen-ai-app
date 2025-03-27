import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
from medmnist import ChestMNIST
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# ==== Hyperparameters ====
BATCH_SIZE = 32
EPOCHS = 10
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

# ==== Convert to Binary (abnormal vs normal) ====
class BinaryChestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label_tensor = torch.from_numpy(label)
        binary_label = torch.tensor([1.0]) if torch.sum(label_tensor) > 0 else torch.tensor([0.0])
        return image, binary_label

    def __len__(self):
        return len(self.dataset)

# ==== Data Loaders ====
train_loader = DataLoader(BinaryChestDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(BinaryChestDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False)

# ==== Model Setup ====
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 1)

# Load existing weights if available
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("No pretrained model found. Starting from scratch.")

model.to(DEVICE)

# ==== Loss, Optimizer, Scheduler ====
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ==== Training Loop ====
best_loss = float('inf')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item(), acc=(correct / total))

    avg_loss = running_loss / len(train_loader)
    avg_acc = correct / total
    scheduler.step()

    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(float))

    print(f"\n Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Avg Acc = {avg_acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f" Saved new best model (loss: {avg_loss:.4f})")

print(f"Final model saved to {MODEL_PATH}")