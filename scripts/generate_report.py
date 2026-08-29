#!/usr/bin/env python3
"""Generate a final evaluation report and learning curves after training."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

def main() -> None:
    history_file = Path("models/logs/training_history.json")
    if not history_file.exists():
        print(f"Error: {history_file} not found. Ensure training has completed and logged history.")
        return
        
    with open(history_file, "r") as f:
        history = json.load(f)
        
    if not history:
        print("Error: training_history.json is empty.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_auc = [h.get("val_auc", 0.0) for h in history]
    
    reports_dir = Path("models/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "loss_curve.png", dpi=300)
    plt.close()
    
    # 2. Accuracy & AUC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.plot(epochs, val_auc, label='Validation AUC', marker='o', linestyle='--')
    plt.title('Training/Validation Accuracy and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / "accuracy_auc_curve.png", dpi=300)
    plt.close()
    
    # Best metrics
    best_epoch_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    best_epoch = epochs[best_epoch_idx]
    best_val_acc = val_acc[best_epoch_idx]
    best_val_loss = val_loss[best_epoch_idx]
    best_val_auc = val_auc[best_epoch_idx]
    
    # Convergence Assessment
    val_loss_trend = "decreasing" if val_loss[-1] <= val_loss[0] else "increasing"
    train_loss_trend = "decreasing" if train_loss[-1] <= train_loss[0] else "increasing"
    
    assessment = f"- **Validation loss trend:** {val_loss_trend.capitalize()}\n"
    assessment += f"- **Training loss trend:** {train_loss_trend.capitalize()}\n"
    if val_loss_trend == "increasing" and train_loss_trend == "decreasing":
        assessment += "- **Warning:** Potential overfitting detected (Validation loss increasing while training loss decreases).\n"
    else:
        assessment += "- **Status:** No clear signs of immediate overfitting.\n"
        
    md_content = f"""# Final Training Report

## System & Model Configuration
- **Backbone Architecture:** EfficientNet-B0
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4 (with CosineAnnealingLR)
- **FP16 AMP:** Enabled (torch.amp.autocast, GradScaler)
- **Total Epochs Run:** {len(epochs)}

## Peak Performance Table
| Metric | Value |
| --- | --- |
| Best Epoch | {best_epoch} |
| Best Validation Accuracy | {best_val_acc:.4f} |
| Best Validation Loss | {best_val_loss:.4f} |
| Best Validation AUC | {best_val_auc:.4f} |

## Convergence Assessment
{assessment}

## Learning Curves
### Loss Curve
![Loss Curve](loss_curve.png)

### Accuracy & AUC Curve
![Accuracy & AUC Curve](accuracy_auc_curve.png)
"""
    
    report_path = reports_dir / "FINAL_TRAINING_REPORT.md"
    with open(report_path, "w") as f:
        f.write(md_content)
        
    print(f"\nFinal report successfully generated at: {report_path}")

if __name__ == '__main__':
    main()
