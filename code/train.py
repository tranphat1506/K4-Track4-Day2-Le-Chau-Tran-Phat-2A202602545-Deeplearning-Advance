"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F)."""
from __future__ import annotations

import os
import json
import random
import time
import argparse
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import từ file cục bộ
from dataset import load_split, check_split, build_transforms, make_loader, NUM_CLASSES
from model import build_model, param_groups, freeze_backbone, count_params, count_gmacs
from losses import build_criterion, mix_batch, mixed_loss

try:
    from eval import save_predictions, compute_metrics
except ImportError:
    print("Warning: Không tìm thấy file eval.py. Các hàm đánh giá tự động sẽ lưu thô.")

@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                
    sampler: str | None = None        
    mix: str | None = None            
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             
    pred_dir: str = "predictions"     
    # --- chỉ bật ở Bước 4 (chung kết) ---
    save_test_predictions: bool = False

def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"

def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_optimizer(model, cfg: Config):
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)

def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name] = param.detach()

    def update(self, model) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                if param.dtype.is_floating_point:
                    self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.detach()))
                else:
                    self.shadow[name].copy_(param.detach())

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    
    if cfg.init == "frozen":
        model.eval()
        model.get_classifier().train()
    else:
        model.train()
        
    running_loss = 0.0
    for images, targets, _ in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type, enabled=cfg.amp):
            if cfg.mix:
                images, mixed_targets = mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)
                logits = model(images)
                loss = mixed_loss(criterion, logits, mixed_targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)
                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema:
            ema.update(model)
            
        running_loss += loss.item() * images.size(0)
        
    return {"train_loss": running_loss / len(loader.dataset), "lr": scheduler.get_last_lr()[0]}

def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits = []
    all_targets = []
    all_filenames = []
    running_loss = 0.0
    
    with torch.inference_mode():
        for images, targets, filenames in tqdm(loader, desc="Eval", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, targets)
                
            running_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_filenames.extend(filenames)
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_filenames, all_targets, all_logits, running_loss / len(loader.dataset)

def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='k')
    ax1.tick_params('y', colors='k')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, 'g--', label='Val Macro-F1')
    ax2.set_ylabel('Macro-F1', color='g')
    ax2.tick_params('y', colors='g')
    
    fig.tight_layout()
    plt.title(title)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.savefig(path, dpi=150)
    plt.close()

def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)
    
    with open(rdir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
        
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)
    
    train_tf = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = build_transforms(train=False, img_size=cfg.img_size)
    
    train_loader = make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg.backbone, num_classes=NUM_CLASSES, drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)
    
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma)
    eval_criterion = build_criterion("ce")
    
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    
    history = []
    best_f1 = -1.0
    best_epoch = -1
    train_times = []
    
    for epoch in range(1, cfg.epochs + 1):
        start_t = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_epoch = time.time() - start_t
        train_times.append(t_epoch)
        
        eval_model = model
        temp_model = None
        if ema:
            import copy
            temp_model = copy.deepcopy(model)
            ema.copy_to(eval_model)
            
        filenames, y_true, logits, val_loss = evaluate(eval_model, val_loader, eval_criterion, device)
        
        if ema and temp_model:
            eval_model.load_state_dict(temp_model.state_dict())
            
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        y_pred = np.argmax(probs, axis=-1)
        
        try:
            from sklearn.metrics import f1_score
            val_f1 = f1_score(y_true, y_pred, average="macro")
        except ImportError:
            val_f1 = 0.0
            
        print(f"Epoch {epoch}/{cfg.epochs} - Train Loss: {train_res['train_loss']:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f} - Time: {t_epoch:.1f}s")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": float(val_f1),
            "lr": train_res["lr"],
            "time": t_epoch
        })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), rdir / "best_model.pth")
            
    pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png", title=f"{cfg.exp_id} - {cfg.backbone}")
    
    model.load_state_dict(torch.load(rdir / "best_model.pth"))
    if ema:
        ema.copy_to(model)
        
    filenames, y_true, logits, _ = evaluate(model, val_loader, eval_criterion, device)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    try:
        save_predictions(pred_path(cfg, "val"), filenames, y_true, probs)
    except NameError:
        df_pred = pd.DataFrame({"Filename": filenames, "y_true": y_true, "y_pred": np.argmax(probs, axis=1)})
        for c in range(9): df_pred[f"p{c}"] = probs[:, c]
        df_pred.to_csv(pred_path(cfg, "val"), index=False)
        
    if cfg.save_test_predictions:
        test_tf = build_transforms(train=False, img_size=cfg.img_size)
        test_df = pd.read_csv(Path(cfg.labels_dir) / f"test_subset{cfg.fold}.csv")
        test_loader = make_loader(test_df, cfg.images_dir, test_tf, cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers)
        
        test_filenames, test_y_true, test_logits, _ = evaluate(model, test_loader, eval_criterion, device)
        test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
        
        try:
            save_predictions(pred_path(cfg, "test"), test_filenames, test_y_true, test_probs)
        except NameError:
            df_pred = pd.DataFrame({"Filename": test_filenames, "y_true": test_y_true, "y_pred": np.argmax(test_probs, axis=1)})
            for c in range(9): df_pred[f"p{c}"] = test_probs[:, c]
            df_pred.to_csv(pred_path(cfg, "test"), index=False)
            
    return {
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "time_per_epoch": np.mean(train_times),
        "params_m": count_params(model),
        "gmacs": count_gmacs(model, cfg.img_size)
    }

def parse_overrides(pairs: list[str]) -> dict:
    d = {}
    config_fields = {f.name: f for f in fields(Config)}
    
    for pair in pairs:
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        if k not in config_fields:
            raise ValueError(f"Unknown config key: {k}")
            
        ftype = config_fields[k].type
        if ftype == int: d[k] = int(v)
        elif ftype == float: d[k] = float(v)
        elif ftype == bool: d[k] = v.lower() in ("true", "1", "yes")
        elif "None" in str(ftype) and v.lower() == "none": d[k] = None
        else: d[k] = v
    return d

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè cấu hình, ví dụ: exp_id=B01 batch_size=32")
    args = parser.parse_args()
    
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"Running config: {cfg}")
    
    res = run(cfg)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()
