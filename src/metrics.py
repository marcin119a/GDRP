import torch
import numpy as np
from scipy.stats import pearsonr

def pearson_corr(x, y):
    """
    Calculate Pearson correlation coefficient between x and y.
    Handles both torch tensors and numpy arrays.
    """
    # Convert to numpy if needed
    if torch.is_tensor(x):
        x = x.detach().numpy()
    if torch.is_tensor(y):
        y = y.detach().numpy()
    
    # Ensure 1D arrays
    x = x.flatten()
    y = y.flatten()
    
    # Check for valid data
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x={len(x)}, y={len(y)}")
    
    if len(x) == 0:
        return 0.0
    
    # Check for constant values
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    # Calculate correlation using scipy for better numerical stability
    try:
        correlation, _ = pearsonr(x, y)
        return correlation
    except Exception as e:
        print(f"Error calculating Pearson correlation: {e}")
        return 0.0

def spearman_corr(x, y):
    """
    Calculate Spearman correlation coefficient between x and y.
    Handles both torch tensors and numpy arrays.
    """
    from scipy.stats import spearmanr
    
    # Convert to numpy if needed
    if torch.is_tensor(x):
        x = x.detach().numpy()
    if torch.is_tensor(y):
        y = y.detach().numpy()
    
    # Ensure 1D arrays
    x = x.flatten()
    y = y.flatten()
    
    # Check for valid data
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x={len(x)}, y={len(y)}")
    
    if len(x) == 0:
        return 0.0
    
    # Calculate correlation using scipy
    try:
        correlation, _ = spearmanr(x, y)
        return correlation
    except Exception as e:
        print(f"Error calculating Spearman correlation: {e}")
        return 0.0