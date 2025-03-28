from dataclasses import dataclass, field
from torchvision import models, transforms
import torch.nn as nn
import torch
import torchvision
import os
from PIL import Image

CHEST_MNIST_LABELS = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule",
    "pneumonia", "pneumothorax", "consolidation", "edema", "emphysema",
    "fibrosis", "pleural thickening", "hernia"
]

@dataclass
class FractureModel:
    model_path: str
    labels: list[str] = field(default_factory=lambda: CHEST_MNIST_LABELS)
    device: torch.device = field(default_factory=lambda: (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ))
    model: nn.Module = field(init=False)
    transform: transforms.Compose = field(init=False)

    def __post_init__(self):
        print(f"📦 Loading model from: {self.model_path}")
        self._build_model()
        self._load_model()
        self._build_transform()

    def _build_model(self):
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.labels))  # Multi-label output
        self.model = model.to(self.device).eval()

    def _load_model(self):
        try:
            state = torch.load(self.model_path, map_location=self.device)
            incompatible_keys = self.model.load_state_dict(state, strict=False)
            print("✅Model loaded successfully.")

            if incompatible_keys.missing_keys:
                print(f"⚠️ Missing keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                print(f"⚠️ Unexpected keys: {incompatible_keys.unexpected_keys}")

        except Exception as e:
            print("Failed to load model:", e)
            raise RuntimeError("Model could not be loaded. Check path or structure.")

    def _build_transform(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def predict(self, image: Image.Image, threshold: float = 0.5):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.sigmoid(output)[0]  # Shape: [14]
            predictions = {
                label: float(prob)
                for label, prob in zip(self.labels, probs)
                if prob >= threshold
            }
        return predictions
