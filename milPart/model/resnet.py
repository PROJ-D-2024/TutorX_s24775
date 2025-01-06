import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Custom Dataset for Spectrogram Images
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                self.samples.append((file_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        image = Image.open(file_path).convert("RGB")  # Load image
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for Spectrogram Images
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize to ResNet input size
    transforms.ToTensor(),         # Convert image to tensor
])

# Dataset and DataLoader
train_dataset = SpectrogramDataset(root_dir="../../data/spectrogramTRAIN", transform=transform)
val_dataset = SpectrogramDataset(root_dir="../../data/spectrogramTRAIN", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pretrained ResNet Model
resnet = models.resnet18(pretrained=True)
num_features = resnet.fc.in_features  # Number of input features for the final layer

# Modify the Fully Connected Layer for Your Task
num_classes = len(train_dataset.classes)
resnet.fc = nn.Linear(num_features, num_classes)

# Move the Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward Pass
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validation Step
    resnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the Fine-Tuned Model
torch.save(resnet.state_dict(), "resnet_spectrogram.pth")
