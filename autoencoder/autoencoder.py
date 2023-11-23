"""Autoencoder to autoencode images (128*128) to detect anomalies"""
import torch.nn as nn

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    """Autoencoder class"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7001316, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 7001316),
            nn.ReLU(),
            nn.Unflatten(1, (196, 189, 189)),
            nn.ReLU(),
            nn.ConvTranspose2d(196, 128, kernel_size=10, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=10, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass of the autoencoder"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
