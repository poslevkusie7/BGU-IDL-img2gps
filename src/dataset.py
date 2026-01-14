from torchvision import transforms

def get_transforms(mode='train', img_size=224):
    """
    Campus-specific augmentation strategy.
    
    For building recognition, we CAN use some spatial augmentation
    because buildings are large and identifiable from different angles.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            # 1. Resize first to ensure we keep the building in frame
            transforms.Resize(256),
            
            # 2. Mild random crop (simulates different viewpoints)
            # 224/256 = 87.5% of image, keeps main building visible
            transforms.RandomCrop(img_size, padding=8, padding_mode='reflect'),
            
            # 3. Small rotation (simulates camera tilt, ±5 degrees)
            transforms.RandomRotation(degrees=5),
            
            # 4. Photometric augmentations (IMPORTANT for outdoor scenes)
            transforms.ColorJitter(
                brightness=0.4,    # Different times of day
                contrast=0.4,      # Cloudy vs sunny
                saturation=0.3,    # Color variation
                hue=0.05          # Slight color shift
            ),
            
            # 5. Random perspective (simulates walking at different angles)
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            
            # 6. Lighting variations
            transforms.RandomGrayscale(p=0.05),
            
            # 7. Weather simulation
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            
            # 8. Convert to tensor
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            
            # 9. Random erasing (simulates occlusions like people, trees)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
        
    else:  # val/test - use center crop for consistency
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])