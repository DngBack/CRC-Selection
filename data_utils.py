"""
Data utilities for CRC-Select: splitting train/calibration/test
"""

import numpy as np
from typing import Tuple


def split_train_calibration(x_train: np.ndarray, 
                            y_train: np.ndarray,
                            cal_ratio: float = 0.2,
                            seed: int = 42) -> Tuple:
    """
    Split training data into train and calibration sets.
    
    Args:
        x_train: training images [N, H, W, C]
        y_train: training labels [N, num_classes + 1]
        cal_ratio: fraction for calibration (default 0.2 = 20%)
        seed: random seed
    
    Returns:
        (x_train_new, y_train_new, x_cal, y_cal)
    """
    np.random.seed(seed)
    
    n_total = len(x_train)
    n_cal = int(n_total * cal_ratio)
    
    # Random permutation
    indices = np.random.permutation(n_total)
    cal_indices = indices[:n_cal]
    train_indices = indices[n_cal:]
    
    x_cal = x_train[cal_indices]
    y_cal = y_train[cal_indices]
    x_train_new = x_train[train_indices]
    y_train_new = y_train[train_indices]
    
    print(f"Data split: Train={len(x_train_new)}, Calibration={len(x_cal)}")
    
    return x_train_new, y_train_new, x_cal, y_cal


def load_ood_dataset(ood_name: str, normalize_stats: Tuple = None):
    """
    Load OOD dataset for evaluation.
    
    Args:
        ood_name: 'svhn', 'cifar100', 'textures', etc.
        normalize_stats: (mean, std) from ID dataset for normalization
    
    Returns:
        x_ood: OOD images
    """
    if ood_name == 'svhn':
        # Load SVHN test set
        import scipy.io as spio
        try:
            mat = spio.loadmat('datasets/test_32x32.mat', squeeze_me=True)
            x_ood = mat["X"]
            x_ood = np.moveaxis(x_ood, -1, 0)
            x_ood = x_ood.astype('float32')
            
            # Normalize if stats provided
            if normalize_stats is not None:
                mean, std = normalize_stats
                x_ood = (x_ood - mean) / (std + 1e-7)
            
            print(f"Loaded SVHN OOD: {len(x_ood)} samples")
            return x_ood
        except FileNotFoundError:
            print(f"SVHN dataset not found. Please download test_32x32.mat to datasets/")
            return None
    
    elif ood_name == 'cifar100':
        # Load CIFAR-100 as OOD for CIFAR-10
        try:
            from keras.datasets import cifar100
            (_, _), (x_ood, _) = cifar100.load_data()
            x_ood = x_ood.astype('float32')
            
            if normalize_stats is not None:
                mean, std = normalize_stats
                x_ood = (x_ood - mean) / (std + 1e-7)
            
            print(f"Loaded CIFAR-100 OOD: {len(x_ood)} samples")
            return x_ood
        except Exception as e:
            print(f"Error loading CIFAR-100: {e}")
            return None
    
    else:
        print(f"Unknown OOD dataset: {ood_name}")
        return None


def create_mixture_dataset(x_id: np.ndarray,
                          y_id: np.ndarray, 
                          x_ood: np.ndarray,
                          ood_ratio: float = 0.1,
                          seed: int = 42):
    """
    Create ID + OOD mixture for testing.
    
    Args:
        x_id: ID test images
        y_id: ID test labels
        x_ood: OOD images
        ood_ratio: fraction of OOD (e.g., 0.1 = 10% OOD)
        seed: random seed
    
    Returns:
        (x_mix, y_mix, is_ood_mask)
        y_mix has dummy labels for OOD (last class)
        is_ood_mask: boolean array indicating OOD samples
    """
    np.random.seed(seed)
    
    n_id = len(x_id)
    n_ood_target = int(n_id * ood_ratio / (1 - ood_ratio))
    n_ood_target = min(n_ood_target, len(x_ood))
    
    # Sample OOD
    ood_indices = np.random.choice(len(x_ood), n_ood_target, replace=False)
    x_ood_sample = x_ood[ood_indices]
    
    # Create dummy labels for OOD (use last class as "OOD" class)
    num_classes = y_id.shape[1]
    y_ood_dummy = np.zeros((n_ood_target, num_classes))
    y_ood_dummy[:, -1] = 1  # Mark as OOD class
    
    # Concatenate
    x_mix = np.concatenate([x_id, x_ood_sample], axis=0)
    y_mix = np.concatenate([y_id, y_ood_dummy], axis=0)
    
    # Create mask
    is_ood_mask = np.concatenate([
        np.zeros(n_id, dtype=bool),
        np.ones(n_ood_target, dtype=bool)
    ])
    
    # Shuffle
    shuffle_indices = np.random.permutation(len(x_mix))
    x_mix = x_mix[shuffle_indices]
    y_mix = y_mix[shuffle_indices]
    is_ood_mask = is_ood_mask[shuffle_indices]
    
    print(f"Created mixture: ID={n_id}, OOD={n_ood_target}, ratio={ood_ratio:.1%}")
    
    return x_mix, y_mix, is_ood_mask

