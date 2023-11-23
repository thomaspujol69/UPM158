"""Autoencoder to autoencode images (128*128) to detect anomalies"""
import torch.nn as nn

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    """Autoencoder class"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        num_input_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, 128, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, num_input_channels, kernel_size=5),
            nn.ReLU(True)
        )

    def forward(self, x):
        """Forward pass of the autoencoder"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
