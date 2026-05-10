"""
Policy Network that selects nodes during simulation step in the MCTS process.
"""

import torch

class PolicyNetwork(torch.nn.Module):
    """
    Is used to select the most possible move given some state.
    """
    def __init__(self, in_channels: int, n_classes: int) -> None:
        """
        Parameters:
            in_channels: number of input channels passed to the first convolution.
            n_classes: number of classes at the output of the whole architecture.
        """
        super().__init__()
        # if there are more channels than 1 kernel is in 3D not 2D as usually depicted
        # so convolution is applied to the whole depth at once and then everything is added
        # to produce one scalar value. After stridding through the input each 3D kernel produces
        # 2D feature map. The size of output channels depends on the number of 3D kernels that
        # are at this layer. Mixing input channels is required to detect sth meaningful - it
        # is because channels contain related information.

        # input size = 10
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.bnorm2d1 = torch.nn.BatchNorm2d(8)
        # input size = 10
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bnorm2d2 = torch.nn.BatchNorm2d(16)
        # input size = 10
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bnorm2d3 = torch.nn.BatchNorm2d(32)
        # input size = 10
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bnorm2d4 = torch.nn.BatchNorm2d(64)
        # input size = 10
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bnorm2d5 = torch.nn.BatchNorm2d(128)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = torch.nn.Flatten()

        # Fully connected layers form clasifier that given some abstract
        # information from the CNN, outputs probability over some finite space.
        # Feature Map Size Formula:
        # O = Floor[(I - F + 2P) / S] + 1
        # Where:
        # O - output size
        # Floor - floor operation that ensures result is int
        # I - input size - the height or width of the input feature map
        # F - filter size - the size of the convolutional kernel
        # P - padding size - the number of paddign pixels added to the border
        # S - stride - the size of the step the filter takes when moving across the input
        # Feature maps will be flattened and concatenated in order to be passed to the
        # linear layer.
        self.lnorm1 = torch.nn.LayerNorm(128 * 10 * 10)
        self.fc1 = torch.nn.Linear(128 * 10 * 10, 2048)
        self.lnorm2 = torch.nn.LayerNorm(2048)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.lnorm3 = torch.nn.LayerNorm(1024)
        # for 10x10 board the n_classes will be 50x50 = 2500
        self.fc3 = torch.nn.Linear(1024, n_classes)
        self.relu = torch.nn.ELU()

        self.dropout = torch.nn.Dropout(p=0.2)

        self.softmax = torch.nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            the output tensor after passing through the network.
        """
        x = self.relu(self.bnorm2d1(self.conv1(x)))
        x = self.relu(self.bnorm2d2(self.conv2(x)))
        x = self.relu(self.bnorm2d3(self.conv3(x)))
        x = self.relu(self.bnorm2d4(self.conv4(x)))
        x = self.relu(self.bnorm2d5(self.conv5(x)))
        x = self.flatten(x)
        x = self.lnorm1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.lnorm2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.lnorm3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
