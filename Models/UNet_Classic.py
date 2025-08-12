# """
# Adapted from:

#     Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark for Scientific Machine Learning
#     Source: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py

# If you use this implementation, please cite original work above.
# """

# # %%
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# from torch.utils.checkpoint import checkpoint


# conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
# conv_transpose_modules = {
#     1: nn.ConvTranspose1d,
#     2: nn.ConvTranspose2d,
#     3: nn.ConvTranspose3d,
# }
# pool_modules = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
# norm_modules = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


# class UNetClassic(nn.Module):
#     def __init__(
#         self,
#         dim_in: int,
#         dim_out: int,
#         n_spatial_dims: int,
#         spatial_resolution: tuple[int, ...],
#         init_features: int = 32,
#         gradient_checkpointing: bool = False,
#     ):
#         super().__init__()  # Fixed: removed the incorrect parameters
        
#         # Store the parameters as instance variables
#         self.n_spatial_dims = n_spatial_dims
#         self.spatial_resolution = spatial_resolution
#         self.gradient_checkpointing = gradient_checkpointing
        
#         features = init_features
#         self.encoder1 = self._block(dim_in, features, name="enc1")
#         self.pool1 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
#         self.encoder2 = self._block(features, features * 2, name="enc2")
#         self.pool2 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
#         self.encoder3 = self._block(features * 2, features * 4, name="enc3")
#         self.pool3 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
#         self.encoder4 = self._block(features * 4, features * 8, name="enc4")
#         self.pool4 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)

#         self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = conv_transpose_modules[n_spatial_dims](
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = conv_transpose_modules[n_spatial_dims](
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = conv_transpose_modules[n_spatial_dims](
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = conv_transpose_modules[n_spatial_dims](
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = self._block(features * 2, features, name="dec1")

#         self.conv = conv_modules[n_spatial_dims](  # Fixed: changed from conv_transpose_modules to conv_modules
#             in_channels=features, out_channels=dim_out, kernel_size=1
#         )

#     def optional_checkpointing(self, layer, *inputs, **kwargs):
#         if self.gradient_checkpointing:
#             return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
#         else:
#             return layer(*inputs, **kwargs)

#     def forward(self, x):

#         x = x.squeeze(dim=-1)
#         enc1 = self.optional_checkpointing(self.encoder1, x)
#         enc2 = self.optional_checkpointing(self.encoder2, self.pool1(enc1))
#         enc3 = self.optional_checkpointing(self.encoder3, self.pool2(enc2))
#         enc4 = self.optional_checkpointing(self.encoder4, self.pool3(enc3))

#         bottleneck = self.optional_checkpointing(self.bottleneck, self.pool4(enc4))

#         dec4 = self.optional_checkpointing(self.upconv4, bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.optional_checkpointing(self.decoder4, dec4)
#         dec3 = self.optional_checkpointing(self.upconv3, dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.optional_checkpointing(self.decoder3, dec3)
#         dec2 = self.optional_checkpointing(self.upconv2, dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.optional_checkpointing(self.decoder2, dec2)
#         dec1 = self.optional_checkpointing(self.upconv1, dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.optional_checkpointing(self.decoder1, dec1)
#         return self.conv(dec1).unsqueeze(dim=-1)  # Add back the last dimension

#     def _block(self, in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         conv_modules[self.n_spatial_dims](
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (
#                         name + "norm1",
#                         norm_modules[self.n_spatial_dims](num_features=features),
#                     ),
#                     (name + "tanh1", nn.Tanh()),
#                     (
#                         name + "conv2",
#                         conv_modules[self.n_spatial_dims](
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (
#                         name + "norm2",
#                         norm_modules[self.n_spatial_dims](num_features=features),
#                     ),
#                     (name + "tanh2", nn.Tanh()),
#                 ]
#             )
#         )
#     def count_params(self):
#         """
#         Count the number of trainable parameters in the model.
#         """
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

# # # %%
# # # Example usage:
# # if __name__ == "__main__":
# #     import torch
    
# #     # Example configuration
# #     dim_in = 3  # input channels
# #     dim_out = 1  # output channels
# #     n_spatial_dims = 2  # 2D case
# #     spatial_resolution = (100, 100)  # height, width
# #     init_features = 32
    
# #     # Create the model
# #     model = UNetClassic(
# #         dim_in=dim_in,
# #         dim_out=dim_out,
# #         n_spatial_dims=n_spatial_dims,
# #         spatial_resolution=spatial_resolution,
# #         init_features=init_features
# #     )
    
# #     # Create dummy input
# #     batch_size = 4
# #     dummy_input = torch.randn(batch_size, dim_in, *spatial_resolution)
    
# #     # Forward pass
# #     output = model(dummy_input)
# #     print(f"Input shape: {dummy_input.shape}")
# #     print(f"Output shape: {output.shape}")
# #     print(f"Model created successfully!")



"""
Adapted from:

    Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark for Scientific Machine Learning
    Source: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py

If you use this implementation, please cite original work above.

Modified to handle arbitrary input sizes without shape mismatches.
"""

# %%
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}
pool_modules = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
norm_modules = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
adaptive_pool_modules = {1: nn.AdaptiveMaxPool1d, 2: nn.AdaptiveMaxPool2d, 3: nn.AdaptiveMaxPool3d}


class UNetClassic(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        init_features: int = 32,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        # Store the parameters as instance variables
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution
        self.gradient_checkpointing = gradient_checkpointing
        
        # Calculate target sizes for each pooling level
        self.pool_sizes = self._calculate_pool_sizes(spatial_resolution)
        
        features = init_features
        self.encoder1 = self._block(dim_in, features, name="enc1")
        self.pool1 = adaptive_pool_modules[n_spatial_dims](self.pool_sizes[0])
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = adaptive_pool_modules[n_spatial_dims](self.pool_sizes[1])
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = adaptive_pool_modules[n_spatial_dims](self.pool_sizes[2])
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = adaptive_pool_modules[n_spatial_dims](self.pool_sizes[3])

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        # Use interpolation for upsampling to ensure exact size matching
        self.decoder4 = self._block((features * 8) + (features * 16), features * 8, name="dec4")
        self.decoder3 = self._block((features * 4) + (features * 8), features * 4, name="dec3")
        self.decoder2 = self._block((features * 2) + (features * 4), features * 2, name="dec2")
        self.decoder1 = self._block(features + (features * 2), features, name="dec1")

        self.conv = conv_modules[n_spatial_dims](
            in_channels=features, out_channels=dim_out, kernel_size=1
        )

    def _calculate_pool_sizes(self, spatial_resolution):
        """Calculate progressive pooling sizes to avoid shape mismatches."""
        pool_sizes = []
        current_size = list(spatial_resolution)
        
        for i in range(4):  # 4 pooling levels
            # Calculate next size (roughly half, but ensure it's at least 1)
            next_size = [max(1, s // 2) for s in current_size]
            pool_sizes.append(tuple(next_size))
            current_size = next_size
            
        return pool_sizes

    def _upsample_to_match(self, x, target_tensor):
        """Upsample tensor x to match the spatial dimensions of target_tensor."""
        target_size = target_tensor.shape[2:]  # Get spatial dimensions
        
        if self.n_spatial_dims == 1:
            return nn.functional.interpolate(x, size=target_size, mode='linear', align_corners=False)
        elif self.n_spatial_dims == 2:
            return nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        elif self.n_spatial_dims == 3:
            return nn.functional.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {self.n_spatial_dims}")

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x):
        # Remove the last dimension if it exists (for compatibility)
        x = x.squeeze(dim=-1)
        
        # Encoder path
        enc1 = self.optional_checkpointing(self.encoder1, x)
        pool1_out = self.pool1(enc1)
        
        enc2 = self.optional_checkpointing(self.encoder2, pool1_out)
        pool2_out = self.pool2(enc2)
        
        enc3 = self.optional_checkpointing(self.encoder3, pool2_out)
        pool3_out = self.pool3(enc3)
        
        enc4 = self.optional_checkpointing(self.encoder4, pool3_out)
        pool4_out = self.pool4(enc4)

        bottleneck = self.optional_checkpointing(self.bottleneck, pool4_out)

        # Decoder path with careful size matching
        dec4 = self._upsample_to_match(bottleneck, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.optional_checkpointing(self.decoder4, dec4)
        
        dec3 = self._upsample_to_match(dec4, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.optional_checkpointing(self.decoder3, dec3)
        
        dec2 = self._upsample_to_match(dec3, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.optional_checkpointing(self.decoder2, dec2)
        
        dec1 = self._upsample_to_match(dec2, enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.optional_checkpointing(self.decoder1, dec1)
        
        output = self.conv(dec1).unsqueeze(dim=-1)  # Final output

            
        return output

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        conv_modules[self.n_spatial_dims](
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "norm1",
                        norm_modules[self.n_spatial_dims](num_features=features),
                    ),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        conv_modules[self.n_spatial_dims](
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "norm2",
                        norm_modules[self.n_spatial_dims](num_features=features),
                    ),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    
    def count_params(self):
        """
        Count the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# # %%
# # Example usage and testing with various input sizes:
# if __name__ == "__main__":
#     import torch
    
#     def test_model_with_size(spatial_resolution, n_spatial_dims=2):
#         """Test the model with a specific input size."""
#         print(f"\nTesting with spatial resolution: {spatial_resolution}")
        
#         # Example configuration
#         dim_in = 3  # input channels
#         dim_out = 1  # output channels
#         init_features = 32
        
#         # Create the model
#         model = UNetClassic(
#             dim_in=dim_in,
#             dim_out=dim_out,
#             n_spatial_dims=n_spatial_dims,
#             spatial_resolution=spatial_resolution,
#             init_features=init_features
#         )
        
#         # Create dummy input
#         batch_size = 2
#         dummy_input = torch.randn(batch_size, dim_in, *spatial_resolution)
        
#         try:
#             # Forward pass
#             with torch.no_grad():
#                 output = model(dummy_input)
            
#             print(f"✓ Input shape: {dummy_input.shape}")
#             print(f"✓ Output shape: {output.shape}")
#             print(f"✓ Success! Parameters: {model.count_params():,}")
            
#             # Verify output spatial dimensions match input
#             assert output.shape[2:] == dummy_input.shape[2:], f"Shape mismatch: {output.shape[2:]} != {dummy_input.shape[2:]}"
#             print("✓ Spatial dimensions preserved correctly")
            
#         except Exception as e:
#             print(f"✗ Error: {e}")
#             return False
        
#         return True
    
#     # Test with various input sizes that commonly cause issues
#     test_cases = [
#         (100, 100),    # Original case
#         (128, 128),    # Power of 2
#         (96, 96),      # Multiple of 16
#         (101, 97),     # Odd numbers
#         (200, 150),    # Different aspect ratio
#         (33, 33),      # Small odd size
#         (64, 128),     # Different dimensions
#     ]
    
#     print("Testing U-Net with various input sizes...")
    
#     for case in test_cases:
#         success = test_model_with_size(case)
#         if not success:
#             print(f"Failed for case: {case}")
#             break
#     else:
#         print("\n🎉 All test cases passed! The model handles arbitrary input sizes correctly.")
        
#     # Test 1D case
#     print("\n" + "="*50)
#     print("Testing 1D case:")
#     test_model_with_size((256,), n_spatial_dims=1)
    
#     # Test 3D case  
#     print("\n" + "="*50)
#     print("Testing 3D case:")
#     test_model_with_size((32, 32, 32), n_spatial_dims=3)