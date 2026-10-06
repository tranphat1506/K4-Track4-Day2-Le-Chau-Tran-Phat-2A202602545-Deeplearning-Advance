"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}

def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head."""
    # Đóng băng toàn bộ
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở băng (requires_grad = True) riêng cho phần head (classifier)
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp."""
    if init == "scratch":
        pretrained = False
        
    model = timm.create_model(
        name, 
        pretrained=pretrained, 
        num_classes=num_classes, 
        drop_rate=drop_rate
    )
    
    if hasattr(model, 'pretrained_cfg'):
        print(f"Loaded weights with tag: {model.pretrained_cfg.get('tag', 'unknown')}")
        
    if init == "frozen":
        freeze_backbone(model)
        
    return model


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm (backbone có decay, backbone không decay, head)."""
    classifier = model.get_classifier()
    head_param_ids = [id(p) for p in classifier.parameters()]
    
    backbone_decay = []
    backbone_no_decay = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if id(param) in head_param_ids:
            head_params.append(param)
        else:
            # ndim <= 1 thường là norm layers (BatchNorm, LayerNorm weights) hoặc bias
            if param.ndim <= 1 or name.endswith(".bias"):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
                
    groups = [
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, img_size, img_size)
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
        return macs / 1e9
    except ImportError:
        print("LƯU Ý: Vui lòng cài thop để đếm GMAC (chạy lệnh: !pip install thop)")
        return 0.0
