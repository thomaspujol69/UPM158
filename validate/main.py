"""V"""

import os
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from autoencoder import Autoencoder
from dataset import Dataset

# let's test the model with the image /data/val/normal/normal_1.jpg
IMAGE_PATH = "/data/val/normal/normal_1.jpg"

model = Autoencoder()

# Define transform
RESIZE = 196
transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
])

# Load dataset
val_dataset = Dataset(split='val', transform=transform)
# Define the dataloader
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=128,
                                        shuffle=True)

# Load the model
model.load_state_dict(torch.load("autoencoder/models/2023-11-23_21-42-40.pth"))
model.eval()

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Exectution device: {device}")

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Test the autoencoder
i=0
while os.path.exists(f"dataset/data/val_output/{i}_input.jpg"):
    i+=1
with torch.no_grad():
    for data in val_loader:
        img, _ = data
        image = img[0].cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(f"dataset/data/val_output/{i}_input.jpg")
        img = img.to(device)
        output = model(img)
        loss = criterion(output, img)
        print(f"Loss: {loss.item():.4f}")

        # Save the image
        output = output.cpu().detach().numpy()
        output = output[0].transpose(1, 2, 0)
        output = Image.fromarray((output * 255).astype(np.uint8))
        output.save(f"dataset/data/val_output/{i}_output.jpg")
        i+=1
