import torch
import torch.nn as nn
import torch.nn.functional as F


class Periodic3DCNN(nn.Module):
    """Expected input shape: (N, C, D, H, W) with H = W = 121, D = 4."""

    def __init__(self, in_channels=7, out_features=3, activation_fn=nn.ReLU):
        super(Periodic3DCNN, self).__init__()

        # Set the desired activation function
        self.activation = activation_fn()

        # --- Convolutional Backbone ---
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv4 = nn.Conv3d(64, 128, kernel_size=(3, 4, 4), padding=0)
        self.bn4 = nn.BatchNorm3d(128)

        # --- Probabilistic Head ---
        self.flattened_size = 128 * 4 * 10 * 10
        # self.flattened_size = 64 * 4 * 13 * 13
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc_mu = nn.Linear(512, out_features)
        self.fc_log_var = nn.Linear(512, out_features)

    def reparameterize(self, mu, log_var):
        # Calculate std from log_var for numerical stability
        std = torch.exp(0.5 * log_var)
        # Generate random noise from a standard normal distribution
        epsilon = torch.randn_like(std)
        # Return the sampled output
        return mu + std * epsilon

    def forward(self, x):

        # This pads ONLY the Depth dimension (now at index 2) circularly.
        padding_instruction = (0, 0, 0, 0, 1, 1)
        x = F.pad(x, padding_instruction, mode="circular")

        # --- Conv Block 1 ---
        x = self.conv1(x)  # -> (N, 16, 4, 119, 119)
        x = self.activation(self.bn1(x))
        x = self.pool1(x)  # -> (N, 16, 4, 59, 59)

        # --- Conv Block 2 ---
        x = F.pad(x, padding_instruction, mode="circular")
        x = self.conv2(x)  # -> (N, 32, 4, 57, 57)
        x = self.activation(self.bn2(x))
        x = self.pool2(x)  # -> (N, 32, 4, 28, 28)

        # --- Conv Block 3 ---
        x = F.pad(x, padding_instruction, mode="circular")
        x = self.conv3(x)  # -> (N, 64, 4, 26, 26)
        x = self.activation(self.bn3(x))
        x = self.pool3(x)  # -> (N, 64, 4, 13, 13)

        # --- Conv Block 4 ---
        x = F.pad(x, padding_instruction, mode="circular")
        x = self.conv4(x)  # -> (N, 128, 4, 10, 10)
        x = self.activation(self.bn4(x))

        # --- Flatten and predict distribution ---
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))  # Use the togglable activation
        x = self.dropout(x)

        # Predict the mean and log variance of the output distribution
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        # Get a sample from the distribution using the reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Return all three for use in a specialized loss function
        return mu, log_var, z
