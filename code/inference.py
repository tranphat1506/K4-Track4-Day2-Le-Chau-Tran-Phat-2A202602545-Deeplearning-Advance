"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from tqdm.auto import tqdm


def predict_logits(model, loader, device, view=None):
    model.eval()
    all_logits = []
    all_targets = []
    all_filenames = []
    
    with torch.inference_mode():
        for images, targets, filenames in tqdm(loader, desc="Inference", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            if view is not None:
                images = view(images)
                
            with autocast(device_type=device.type):
                logits = model(images)
                
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_filenames.extend(filenames)
            
    return all_filenames, np.concatenate(all_targets), np.concatenate(all_logits)


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int):
    # Trả về 5 crop: top-left, top-right, bottom-left, bottom-right, center
    b, c, h, w = x.shape
    crops = []
    crops.append(x[..., :crop, :crop])
    crops.append(x[..., :crop, -crop:])
    crops.append(x[..., -crop:, :crop])
    crops.append(x[..., -crop:, -crop:])
    
    start_h = (h - crop) // 2
    start_w = (w - crop) // 2
    crops.append(x[..., start_h:start_h+crop, start_w:start_w+crop])
    return crops


def views_multiscale(x, sizes):
    scaled_views = []
    for s in sizes:
        scaled_views.append(F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False))
    return scaled_views


def aggregate_views(logits_per_view, space: str = "prob"):
    logits_per_view = np.array(logits_per_view) # shape (K, N, 9)
    if space == "prob":
        probs = np.exp(logits_per_view) / np.sum(np.exp(logits_per_view), axis=-1, keepdims=True)
        return np.mean(probs, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return np.exp(avg_logits) / np.sum(np.exp(avg_logits), axis=-1, keepdims=True)
    else:
        raise ValueError("space must be 'prob' or 'logit'")


def ensemble_probs(list_of_probs):
    probs = np.array(list_of_probs)
    return np.mean(probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    # LBFGS tối ưu log_T để giảm Expected Calibration Error (ECE) thông qua NLL Loss
    val_logits = torch.tensor(val_logits, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / torch.exp(log_T), val_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    return torch.exp(log_T).item()


def apply_temperature(logits, T: float):
    logits = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits / T, dim=-1).numpy()


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước để tăng tốc độ suy luận."""
    model.eval()
    
    def _fuse(module):
        last_conv = None
        last_conv_name = None
        
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                last_conv = child
                last_conv_name = name
            elif isinstance(child, nn.BatchNorm2d) and last_conv is not None:
                # Tạo một Conv mới đã gộp
                fused_conv = nn.Conv2d(
                    last_conv.in_channels, last_conv.out_channels, kernel_size=last_conv.kernel_size,
                    stride=last_conv.stride, padding=last_conv.padding, dilation=last_conv.dilation,
                    groups=last_conv.groups, bias=True, padding_mode=last_conv.padding_mode
                )
                
                # Tính trọng số gộp
                w_conv = last_conv.weight.clone().view(last_conv.out_channels, -1)
                w_bn = torch.diag(child.weight.div(torch.sqrt(child.eps + child.running_var)))
                fused_conv.weight.data = torch.mm(w_bn, w_conv).view(fused_conv.weight.size())
                
                # Tính Bias gộp
                b_conv = last_conv.bias if last_conv.bias is not None else torch.zeros(last_conv.weight.size(0))
                b_bn = child.bias - child.weight.mul(child.running_mean).div(torch.sqrt(child.running_var + child.eps))
                fused_conv.bias.data = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1) + b_bn
                
                # Thay thế lớp cũ bằng lớp Identity, lớp Conv cũ bằng lớp Conv mới (gộp BN)
                setattr(module, last_conv_name, fused_conv)
                setattr(module, name, nn.Identity())
                
                last_conv = None
                last_conv_name = None
            else:
                _fuse(child)
                # Đặt lại biến theo dõi nếu chèn lớp khác giữa Conv và BN (ngoại trừ hàm kích hoạt Identity, ReLU)
                if not isinstance(child, (nn.Identity, nn.ReLU, nn.LeakyReLU, nn.Hardswish, nn.Hardsigmoid, nn.GELU, nn.SiLU)):
                    last_conv = None
                    last_conv_name = None
                    
    _fuse(model)
    return model
