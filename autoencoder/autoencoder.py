"""Autoencoder to autoencode images (512x512) to detect anomalies"""
import torch.nn as nn

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    """Autoencoder class"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512*512, 128*128),
            nn.ReLU(),
            nn.Conv2d(128*128, 128*128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128*128, 64*64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(64*64, 32*32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32*32, 64*64),
            nn.ReLU(),
            nn.Conv2d(64*64, 128*128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128*128, 128*128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128*128, 512*512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass of the autoencoder"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
