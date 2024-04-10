import torch
import torch.nn as nn
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, 
                 use_batch_norm: bool = True, use_max_pool: bool = True, dropout: float = 0.7):
        super(ConvBlock, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.use_max_pool = use_max_pool
        self.dropout = dropout
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout2d = nn.Dropout2d(p=dropout)
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        if self.use_max_pool:
            x = self.pool(x)
        if self.dropout > 0:
            x = self.dropout2d(x)
        
        return x

# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, out_channels_init_conv: int=16, use_batch_norm: bool=True, dropout: float = 0.7):

        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes to size appropriately the output of your classifier,
        # and if you use the Dropout layer, use the variable "dropout" to indicate how much to use (like nn.Dropout(p=dropout))

        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        self.activation = nn.LeakyReLU(0.2)
        
        in_channels_init_conv = 3
        kernel_size = 3
        padding = 1

        self.conv01 = ConvBlock(in_channels=in_channels_init_conv, out_channels=out_channels_init_conv, kernel_size=kernel_size, 
                                padding=padding, use_batch_norm = False, use_max_pool = True, dropout = dropout)
        self.conv02 = ConvBlock(in_channels=out_channels_init_conv, out_channels=out_channels_init_conv*2, kernel_size=kernel_size, 
                                padding=padding, use_batch_norm = use_batch_norm, use_max_pool = True, dropout = dropout)
        self.conv03 = ConvBlock(in_channels=out_channels_init_conv*2, out_channels=out_channels_init_conv*4, kernel_size=kernel_size, 
                                padding=padding, use_batch_norm = use_batch_norm, use_max_pool = True, dropout = dropout)
        self.conv04 = ConvBlock(in_channels=out_channels_init_conv*4, out_channels=out_channels_init_conv*8, kernel_size=kernel_size, 
                                padding=padding, use_batch_norm = use_batch_norm, use_max_pool = True, dropout = dropout)
        self.conv05 = ConvBlock(in_channels=out_channels_init_conv*8, out_channels=out_channels_init_conv*16, kernel_size=kernel_size, 
                                padding=padding, use_batch_norm = use_batch_norm, use_max_pool = True, dropout = dropout)
        self.conv06 = ConvBlock(in_channels=out_channels_init_conv*16, out_channels=out_channels_init_conv*32, kernel_size=kernel_size, 
                                padding=padding, use_batch_norm = use_batch_norm, use_max_pool = True, dropout = dropout)
        
        self.flatten = nn.Flatten()
        
        out_channels_conv = out_channels_init_conv*32
        image_dim_conv = 224 // (2 ** 6) # input image size is 224x224, image dim is halved at each conv. stage
        in_channels_init_lin = out_channels_conv * image_dim_conv * image_dim_conv
        
        self.fc_final = nn.Linear(in_channels_init_lin, num_classes)
        
    def forward(self, x):

        # Process the input tensor through the feature extractor,
        # the pooling and the final linear layers (if appropriate for the architecture chosen)

        x = self.conv01(x) # 3x224x224 --> Nx112x112
        x = self.conv02(x) # Nx112x112 --> (2N)x56x56
        x = self.conv03(x) # (2N)x56x56 --> (4N)x28x28
        x = self.conv04(x) # (4N)x28x28 --> (8N)x14x14
        x = self.conv05(x) # (8N)x14x14 --> (16N)x7x7
        x = self.conv06(x) # (16N)x7x7 --> (32N)x3x3
        
        x = self.flatten(x)
        
        x = self.activation(self.fc_final(x)) # (32N)x3x3 --> num_classes

        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)

def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, out_channels_init_conv=16, use_batch_norm=True, dropout=0.3)
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    out = model(images)
    assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"
