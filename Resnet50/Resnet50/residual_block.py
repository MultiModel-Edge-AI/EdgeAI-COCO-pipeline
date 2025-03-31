import torch
import torch.nn as nn

# Define the Residual Block
class Block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, stride=1):
        super(Block, self).__init__()

        # Building the bottleneck block
        self.building_block = nn.Sequential(

            # 1x1 convolution
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),

            # 3x3 convolution
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            
            #1x1 convolution
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels) 
        )

        # Downsample for residual connection if dimensions differ
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)  # ReLU after adding identity

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.building_block(x)
        out += identity       # Add skip connection
        out = self.relu(out)  # Apply ReLU after adding residual
        return out