"""Main module for training dataset"""

import torch
import tqdm
import datetime
import torch.optim as optim
import torchvision.transforms as transforms
from autoencoder import Autoencoder
from dataset import Dataset

# Initialize the autoencoder
model = Autoencoder()

# Define transform
transform = transforms.Compose([
    transforms.Resize((4096, 4096)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = Dataset(split='train',
                                transform=transform)
test_dataset = Dataset(split='test',
                                transform=transform)
# Define the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=128,
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=128)

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 5
pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    pbar.set_description(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), f"autoencoder/models_{datetime.datetime.now()}.pth")
