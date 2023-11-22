"""Autoencoder to autoencode images (128*128) to detect anomalies"""
import torch.nn as nn

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    """Autoencoder class"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128*128*3, 64*64),
            nn.ReLU(),
            nn.Conv2d(64*64, 64*64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64*64, 32*32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(32*32, 16*16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16*16, 32*32),
            nn.ReLU(),
            nn.Conv2d(32*32, 64*64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64*64, 64*64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64*64, 128*128*3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass of the autoencoder"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
