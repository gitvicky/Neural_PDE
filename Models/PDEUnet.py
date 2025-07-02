# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    """
    2D U-Net for Neural PDE Solving
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output channels/classes
        bilinear (bool): Use bilinear upsampling instead of transposed convolutions
        base_channels (int): Base number of channels (doubles at each level)
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_channels=64):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder path
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder path
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits

# Example usage for PDE solving
class PDEUNet(nn.Module):
    """
    U-Net specifically configured for PDE solving with additional features
    """
    def __init__(self, 
                 spatial_channels=2,  # x, y coordinates
                 field_channels=1,    # solution field(s)
                 output_channels=1,   # predicted field(s)
                 base_channels=64,
                 len_x=128,
                 len_y=128,
                 include_residual=True,
                 device='cpu'):
        super(PDEUNet, self).__init__()
        
        input_channels = spatial_channels + field_channels
        self.include_residual = include_residual
        
        self.unet = UNet2D(
            n_channels=input_channels,
            n_classes=output_channels,
            base_channels=base_channels
        )

        self.coords = self.create_coordinate_grid(len_x, len_y)  # Create coordinate grid
        self.coords = self.coords.to(device)

    # Utility function to create coordinate grids
    def create_coordinate_grid(self, len_x, len_y):
        """Create normalized coordinate grid for PDE solving"""
        x_coords = torch.linspace(-1, 1, len_x)
        y_coords = torch.linspace(-1, 1, len_y)

        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        coords = torch.stack([X, Y], dim=0)  # Shape: (2, H, W)
        return coords


    def forward(self, field):
        """
        Args:
            coords: Spatial coordinates (B, 2, H, W) for x, y
            time: Time coordinate (B, 1, H, W) 
            field: Current field values (B, C, H, W)
        """
        # Concatenate all inputs
        coords = self.coords.unsqueeze(0).repeat(field.shape[0], 1, 1, 1)
        x = torch.cat([coords, field], dim=1)
        
        # Pass through U-Net
        output = self.unet(x)
        
        # Optional residual connection
        if self.include_residual and output.shape == field.shape:
            output = output + field
            
        return output

    def count_params(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# # Example of how to use the PDE U-Net
# if __name__ == "__main__":
#     # Example parameters
#     batch_size = 4
#     height, width = 101, 101
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Create model
#     model = PDEUNet(
#         spatial_channels=2,
#         field_channels=1,
#         output_channels=1,
#         len_x=width,
#         len_y=height,
#         base_channels=64,
#         include_residual=True
#     ).to(device)
    
#     field = torch.randn(batch_size, 1, height, width, device=device)  # (B, 1, H, W)
    
#     # Forward pass
#     with torch.no_grad():
#         output = model(field)
#         print(f"Input field shape: {field.shape}")
#         print(f"Output field shape: {output.shape}")
#         print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     print("PDE U-Net created successfully!")


# %% 