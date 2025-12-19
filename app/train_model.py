import os
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision import transforms

from cnn import CNN

class SafeTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)  # convert numpy arrays if needed
        try:
            return self.transform(img)
        except Exception as e:
            print(f"Skipping image due to transform error: {e}")
            # return a black image to keep batch shape consistent
            return Image.new("RGB", (224, 224))

# --- Improved transforms ---
base_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),

    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize([0.5]*3, [0.5]*3),

    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.1),
        ratio=(0.3, 3.3)
    )
])

transform = SafeTransform(base_transform)


# --- Dataset + split ---
dataset = ImageFolder(Path("images"), transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# --- Training setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
cnn = CNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.06)
optimizer = Adam(cnn.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
epochs = 200

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# --- Early stopping ---
early_stop_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

# --- Create folder once at the start ---
date_str = datetime.datetime.now().strftime("%Y%m%d")
time_str = datetime.datetime.now().strftime("%H%M%S")
folder_path = os.path.join("models", date_str + "_" + time_str)
os.makedirs(folder_path, exist_ok=True)

# --- Training loop ---
for epoch in range(epochs):
    cnn.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / train_size
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # --- Validation ---
    cnn.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / val_size
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    # Step scheduler
    scheduler.step(val_epoch_loss)

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc*100:.2f}%")

    # --- Save intermittent model each epoch ---
    model_path = os.path.join(folder_path, f"cnn_epoch_{epoch+1:03d}.pth")
    torch.save(cnn.state_dict(), model_path)

    # Early stopping check
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        # Optionally save best separately too
        best_model_path = os.path.join(folder_path, "cnn_best.pth")
        torch.save(cnn.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# --- Save final metrics plot ---
plot_path = os.path.join(folder_path, "training_metrics.png")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Training metrics saved to {plot_path}")
