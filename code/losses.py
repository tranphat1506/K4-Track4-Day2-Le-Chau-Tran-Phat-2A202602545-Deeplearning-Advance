"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unknown kind of loss: {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, targets):
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        log_pt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_pt)
        
        log_pt_target = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt_target = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        loss = - (1 - pt_target) ** self.gamma * log_pt_target
        
        if self.alpha is not None:
            alpha_target = self.alpha.to(targets.device).gather(0, targets)
            loss = loss * alpha_target
            
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = np.array(counts)
    if beta == 0.0:
        w = 1.0 / counts
        w = w / np.mean(w) 
    else:
        w = (1.0 - beta) / (1.0 - (beta ** counts))
        w = w / np.sum(w) * len(counts) 
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[index, :]
    elif mode == "cutmix":
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x_mix = x.clone()
        x_mix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    else:
        raise ValueError(f"Unknown mode {mode}")
        
    return x_mix, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
