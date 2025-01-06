import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.models as models

# Create a toy dataset
class TinyDataset(Dataset):
    def __init__(self):
        # Four random 1-channel 224x224 images and their scalar targets
        self.data = torch.randn(4, 1, 224, 224)  # 4 images, 1 channel
        self.targets = torch.tensor([0.5, 1.0, 0.0, -0.5])  # 4 scalar values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Prepare the dataset and dataloader
dataset = TinyDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the modified ResNet-34 model
class ResNet34Modified(nn.Module):
    def __init__(self):
        super(ResNet34Modified, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)  # No pretrained weights
        self.resnet34.conv1 = nn.Conv2d(1, self.resnet34.conv1.out_channels,
                                        kernel_size=self.resnet34.conv1.kernel_size,
                                        stride=self.resnet34.conv1.stride,
                                        padding=self.resnet34.conv1.padding,
                                        bias=self.resnet34.conv1.bias is not None)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, 1)

    def forward(self, x):
        return self.resnet34(x)

# Initialize the model, loss function, and optimizer
model = ResNet34Modified()
criterion = nn.MSELoss()  # Mean squared error for regression
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs).squeeze()  # Outputs shape: (batch_size,)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# # Check if the model overfits
# model.eval()
# with torch.no_grad():
#     for inputs, targets in dataloader:
#         outputs = model(inputs).squeeze()
#         print("Targets:", targets.numpy())
#         print("Predictions:", outputs.numpy())
