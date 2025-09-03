#!/usr/bin/env python3
"""
Test script to verify the mask functionality in MaskPEFluxKontextPipeline
"""

import torch
import numpy as np
from PIL import Image

def test_mask_functionality():
    """Test the mask processing functionality"""
    
    print("Testing mask functionality...")
    
    # Create dummy inputs
    batch_size = 1
    height, width = 512, 512
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Create dummy image (3 channels, RGB)
    image = torch.rand(batch_size, 3, height, width, dtype=dtype, device=device)
    
    # Create dummy mask (white square in center)
    mask = torch.zeros(batch_size, 1, height, width, dtype=dtype, device=device)
    center_start = height // 4
    center_end = 3 * height // 4
    mask[:, :, center_start:center_end, center_start:center_end] = 1.0  # White square
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask min/max: {mask.min():.3f}/{mask.max():.3f}")
    print(f"White pixels in mask: {(mask > 0.5).sum().item()}")
    
    # Test interpolation functionality
    import torch.nn.functional as F
    
    # Simulate the latent dimensions (downsampled by factor of 16 for VAE + factor of 2 for packing)
    latent_height = height // 16
    latent_width = width // 16
    
    print(f"Latent dimensions: {latent_height}x{latent_width}")
    
    # Test mask resizing
    mask_resized = F.interpolate(
        mask,
        size=(latent_height, latent_width),
        mode='nearest'
    )
    
    print(f"Resized mask shape: {mask_resized.shape}")
    print(f"White pixels in resized mask: {(mask_resized > 0.5).sum().item()}")
    
    # Test binary mask creation
    binary_mask = (mask_resized > 0.5).float()
    mask_flat = binary_mask.view(batch_size, -1)
    
    print(f"Flattened mask shape: {mask_flat.shape}")
    print(f"Number of white pixels per batch: {mask_flat.sum(dim=1)}")
    
    # Test index generation
    latent_image_ids = torch.zeros(latent_height, latent_width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(latent_height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(latent_width)[None, :]
    
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    ).to(device=device, dtype=dtype)
    
    # Set to 1 (image type)
    image_ids[..., 0] = 1
    
    print(f"Image IDs shape: {image_ids.shape}")
    print(f"Initial image IDs first column unique values: {torch.unique(image_ids[..., 0])}")
    
    # Apply mask
    for b in range(batch_size):
        mask_indices = mask_flat[b] == 1
        if mask_indices.any():
            image_ids[mask_indices, 0] = 2
    
    print(f"After masking, image IDs first column unique values: {torch.unique(image_ids[..., 0])}")
    print(f"Number of masked positions: {(image_ids[..., 0] == 2).sum().item()}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_mask_functionality()
