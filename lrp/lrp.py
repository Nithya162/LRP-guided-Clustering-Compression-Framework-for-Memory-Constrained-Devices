import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from models.net import SimpleFCNN, ResNet9
from data_loaders.mnist_loader import get_mnist_loaders
from utils.visualization import analyze_Rmat_statistics
from utils.metrics import evaluate_model

class UnifiedLRP:
    """Unified Layer-wise Relevance Propagation implementation"""
    
    def __init__(self, model: nn.Module, epsilon: float = 1e-7, gamma: float = 0.05):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.activations = OrderedDict()
        self.handles = []
        self.model_layers_ordered: List[Tuple[str, nn.Module]] = []

    def _clear_hooks_and_data(self):
        for handle in self.handles: handle.remove()
        self.handles = []
        self.activations.clear()
        self.model_layers_ordered = []

    def _register_forward_hooks_and_get_layers(self, sample_input: torch.Tensor):
        self._clear_hooks_and_data()
        
        temp_handles = []
        def hook_fn_factory(name: str, module_obj: nn.Module):
            def hook(module: nn.Module, input_val: Tuple[torch.Tensor, ...], output_val: torch.Tensor):
                if input_val and isinstance(input_val[0], torch.Tensor):
                    self.activations[name + "_in"] = input_val[0].detach().clone()
                self.activations[name + "_out"] = output_val.detach().clone()
                if name not in [item[0] for item in self.model_layers_ordered]:
                    self.model_layers_ordered.append((name, module_obj))
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.MaxPool2d, 
                                   nn.AdaptiveAvgPool2d, nn.Flatten, nn.BatchNorm2d, nn.Dropout)):
                temp_handles.append(module.register_forward_hook(hook_fn_factory(name, module)))
        
        with torch.no_grad():
            self.model.eval()
            self.model(sample_input)
        
        for h in temp_handles: h.remove()

    def _propagate_linear_epsilon(self, R_k: torch.Tensor, layer: nn.Linear, act_in_to_layer: torch.Tensor) -> torch.Tensor:
        W = layer.weight.data
        Z = torch.matmul(act_in_to_layer, W.T)
        if layer.bias is not None: Z += layer.bias.data.unsqueeze(0)
        stab_Z = Z + self.epsilon * torch.sign(Z) + 1e-9
        stab_Z[torch.abs(stab_Z) < 1e-9] = torch.sign(stab_Z[torch.abs(stab_Z) < 1e-9]) * self.epsilon + 1e-9
        s_k = R_k / stab_Z
        R_j = act_in_to_layer * torch.matmul(s_k, W)
        return R_j

    def _propagate_conv2d_gamma(self, R_k: torch.Tensor, layer: nn.Conv2d, act_in_to_layer: torch.Tensor) -> torch.Tensor:
        W = layer.weight.data
        W_prime = W + self.gamma * torch.relu(W)
        params = {'stride': layer.stride, 'padding': layer.padding, 'dilation': layer.dilation, 'groups': layer.groups}
        Z_k = F.conv2d(act_in_to_layer, W_prime, bias=None, **params)
        if layer.bias is not None:
            Z_k = Z_k + layer.bias.data.view(1, -1, 1, 1)
            
        stab_Z_k = Z_k + self.epsilon * torch.sign(Z_k) + 1e-9
        stab_Z_k[torch.abs(stab_Z_k) < 1e-9] = torch.sign(stab_Z_k[torch.abs(stab_Z_k) < 1e-9]) * self.epsilon + 1e-9
        s_k = R_k / stab_Z_k
        
        R_j = F.conv_transpose2d(s_k, W_prime, bias=None, stride=params['stride'], padding=params['padding'], 
                                 output_padding=0, groups=params['groups'], dilation=params['dilation'])
        R_j = act_in_to_layer * R_j
        
        if R_j.shape != act_in_to_layer.shape:
            R_j = F.interpolate(R_j, size=act_in_to_layer.shape[2:], mode='bilinear', align_corners=False)
        return R_j

    def _propagate_pooling(self, R_k: torch.Tensor, pool_layer: nn.Module, act_in_to_pool: torch.Tensor, act_out_from_pool: torch.Tensor) -> torch.Tensor:
        if isinstance(pool_layer, nn.Flatten):
            if R_k.numel() == act_in_to_pool.numel():
                R_j = R_k.view(act_in_to_pool.shape)
            else:
                R_j = R_k
                
        elif isinstance(pool_layer, nn.AdaptiveAvgPool2d):
            if act_in_to_pool.dim() == 4 and R_k.dim() == 4:
                h, w = act_in_to_pool.shape[2:]
                num_elements = h * w
                R_j = R_k.expand(-1, -1, h, w) / num_elements
            elif act_in_to_pool.dim() == 4 and R_k.dim() == 2:
                h, w = act_in_to_pool.shape[2:]
                num_elements = h * w
                R_j = R_k.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w) / num_elements
            else:
                R_j = R_k
                
        elif isinstance(pool_layer, nn.MaxPool2d):
            if act_in_to_pool.dim() == 4 and R_k.dim() == 4:
                try:
                    R_k_upsampled = F.interpolate(R_k, size=act_in_to_pool.shape[2:], mode='nearest')
                    pooled_input = F.max_pool2d(act_in_to_pool, 
                                               kernel_size=pool_layer.kernel_size, 
                                               stride=pool_layer.stride, 
                                               padding=pool_layer.padding)
                    norm_factor = F.interpolate(pooled_input + self.epsilon, 
                                              size=act_in_to_pool.shape[2:], 
                                              mode='nearest')
                    is_max = (act_in_to_pool >= norm_factor - self.epsilon*2).float()
                    R_j = R_k_upsampled * is_max * (act_in_to_pool / (norm_factor + self.epsilon))
                except:
                    R_j = F.interpolate(R_k, size=act_in_to_pool.shape[2:], mode='nearest')
            else:
                R_j = R_k
        else:
            R_j = R_k
            
        return R_j

    def compute_lrp_for_batch(self, input_batch: torch.Tensor, target_class_idx_batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        self._register_forward_hooks_and_get_layers(input_batch[:1])
        
        model_output_logits = self.model(input_batch)
        if target_class_idx_batch is None: 
            target_class_idx_batch = torch.argmax(model_output_logits, dim=1)
        
        R_current = torch.zeros_like(model_output_logits)
        for i in range(input_batch.shape[0]): 
            R_current[i, target_class_idx_batch[i]] = model_output_logits[i, target_class_idx_batch[i]]
        
        batch_layer_relevances_at_output = {}

        for name, module in reversed(self.model_layers_ordered):
            batch_layer_relevances_at_output[name] = R_current.clone().detach()
            
            act_in_to_module = self.activations.get(name + "_in")
            act_out_from_module = self.activations.get(name + "_out")

            if act_in_to_module is None and name == self.model_layers_ordered[0][0]:
                act_in_to_module = input_batch.view(input_batch.size(0), -1) if isinstance(module, nn.Linear) and input_batch.dim() > 2 else input_batch
            if act_in_to_module is None: 
                continue

            if isinstance(module, nn.Linear): 
                R_current = self._propagate_linear_epsilon(R_current, module, act_in_to_module)
            elif isinstance(module, nn.Conv2d): 
                R_current = self._propagate_conv2d_gamma(R_current, module, act_in_to_module)
            elif isinstance(module, (nn.ReLU, nn.Dropout, nn.BatchNorm2d)): 
                R_current = R_current
            elif isinstance(module, (nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten)):
                if act_in_to_module is not None and act_out_from_module is not None: 
                    R_current = self._propagate_pooling(R_current, module, act_in_to_module, act_out_from_module)
        
        self._clear_hooks_and_data()
        return batch_layer_relevances_at_output

    def get_target_layer_relevance_matrices(self, data_loader, target_layer_names: List[str], num_batches_for_Rmat: Optional[int] = None) -> Dict[str, np.ndarray]:
        self.model.eval()
        Rmat_accumulator = {name: [] for name in target_layer_names}
        processed_batches = 0
        
        for input_b, target_b in tqdm(data_loader, desc="LRP for Rmat", leave=False):
            if num_batches_for_Rmat is not None and processed_batches >= num_batches_for_Rmat: 
                break
            input_b = input_b.to(next(self.model.parameters()).device)
            
            layer_rels_batch = self.compute_lrp_for_batch(input_b, target_class_idx_batch=None)
            
            for name in target_layer_names:
                if name in layer_rels_batch:
                    rel_tensor = layer_rels_batch[name]
                    if rel_tensor.dim() == 2: 
                        R_batch_norm_first = rel_tensor.cpu().numpy().T
                    elif rel_tensor.dim() == 4: 
                        R_batch_norm_first = rel_tensor.abs().sum(dim=(2,3)).cpu().numpy().T
                    else: 
                        continue
                    Rmat_accumulator[name].append(R_batch_norm_first)
            processed_batches += 1

        final_Rmat = {}
        for name, rmat_list in Rmat_accumulator.items():
            if rmat_list:
                full_rmat = np.hstack(rmat_list)
                abs_m = np.abs(full_rmat)
                mins = np.min(abs_m, axis=1, keepdims=True)
                maxs = np.max(abs_m, axis=1, keepdims=True)
                ranges = maxs - mins
                ranges[ranges < 1e-9] = 1.0
                final_Rmat[name] = (abs_m - mins) / ranges
                print(f"  Layer {name}: Final Rmat shape {final_Rmat[name].shape}")
            else: 
                final_Rmat[name] = np.array([])
        return final_Rmat


def train_and_save_model(model_instance: nn.Module, model_name: str, train_loader, 
                        epochs: int = 5, lr: float = 0.001, force_train: bool = False, 
                        device=None) -> nn.Module:
    """Train and save a model"""
    model_path = os.path.join("models/saved_models", f"{model_name}_mnist_phase1.pth")
    
    if os.path.exists(model_path) and not force_train:
        print(f"Loading pre-trained {model_name} model from {model_path}")
        try:
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print(f"Error loading state_dict for {model_name}: {e}")
            model_instance.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model_instance.to(device)
        return model_instance

    print(f"Training {model_name} model...")
    model_instance.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model_instance.train()
        running_loss = 0.0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_instance(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model_instance.state_dict(), model_path)
    print(f"Saved {model_name} model to {model_path}")
    return model_instance


def run_phase1_lrp(models='both', force_train=False, device=None):
    """Run Phase 1: LRP analysis"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    RESULTS_DIR = "results/lrp_phase1_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    EPOCHS = 5
    NUM_LRP_BATCHES = 10
    
    train_loader, test_loader = get_mnist_loaders(128, 100)
    
    # Process SimpleFCNN
    if models in ['both', 'fc']:
        fc_model = train_and_save_model(SimpleFCNN(), "SimpleFCNN", train_loader, 
                                       epochs=EPOCHS, force_train=force_train, device=device)
        evaluate_model(fc_model, test_loader, "SimpleFCNN", device)
        
        # LRP for SimpleFCNN
        fc_target_layers = ['fc2', 'fc3', 'fc4', 'fc5']
        lrp_analyzer_fc = UnifiedLRP(fc_model.to(device))
        fc_Rmat_dict = lrp_analyzer_fc.get_target_layer_relevance_matrices(test_loader, fc_target_layers, NUM_LRP_BATCHES)
        analyze_Rmat_statistics(fc_Rmat_dict, "SimpleFCNN", RESULTS_DIR)
        
        # Save results
        np.savez(os.path.join(RESULTS_DIR, 'fc_rmat.npz'), **fc_Rmat_dict)
    
    # Process ResNet9
    if models in ['both', 'resnet']:
        resnet_model = train_and_save_model(ResNet9(in_channels=1, num_classes=10), "ResNet9_MNIST", 
                                          train_loader, epochs=EPOCHS, force_train=force_train, device=device)
        evaluate_model(resnet_model, test_loader, "ResNet9", device)
        
        # LRP for ResNet9
        resnet9_target_layers = ['c2.0', 'c3.0', 'c4.0', 'classifier_head.3']
        lrp_analyzer_resnet = UnifiedLRP(resnet_model.to(device))
        resnet_Rmat_dict = lrp_analyzer_resnet.get_target_layer_relevance_matrices(test_loader, resnet9_target_layers, NUM_LRP_BATCHES)
        analyze_Rmat_statistics(resnet_Rmat_dict, "ResNet9", RESULTS_DIR)
        
        # Save results
        np.savez(os.path.join(RESULTS_DIR, 'resnet_rmat.npz'), **resnet_Rmat_dict)
    
    print("\nPhase 1: LRP Relevance Matrix computation complete.")
    print(f"Results saved to {RESULTS_DIR}")