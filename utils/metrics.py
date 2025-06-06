import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader,
                  model_name: str = "Model",
                  device: torch.device = None) -> float:
    """Evaluate model accuracy"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"{model_name} accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def compute_compression_ratio(original_model: nn.Module, 
                            compressed_model: nn.Module) -> float:
    """Compute compression ratio between two models"""
    orig_params = count_parameters(original_model)
    comp_params = count_parameters(compressed_model)
    return 100 * comp_params / orig_params