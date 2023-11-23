"""Main module for training dataset"""

import tqdm
import datetime
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from autoencoder import Autoencoder
from dataset import Dataset

# Initialize the autoencoder
model = Autoencoder()

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = Dataset(split='train', transform=transform)
test_dataset = Dataset(split='test', transform=transform)
# Define the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=128,
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=128)

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Exectution device: {device}")

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
NUM_EPOCHS = 5
pbar = tqdm.tqdm(range(NUM_EPOCHS))
losses = []
for epoch in pbar:
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {loss.item():.4f}")
    losses.append(loss.item())

# Save the model
torch.save(model.state_dict(), open(f"autoencoder/models/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth", 'wb'))
with open(f"autoencoder/models/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", 'w') as f:
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write("Losses: \n")
    for loss in losses:
        f.write(f"    {loss}\n")
    f.write("Layers: \n")
    for layer in model.encoder:
        f.write(f"    {layer}\n")

