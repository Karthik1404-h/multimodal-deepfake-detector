# Final Training Report

## System & Model Configuration
- **Backbone Architecture:** EfficientNet-B0
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4 (with CosineAnnealingLR)
- **FP16 AMP:** Enabled (torch.amp.autocast, GradScaler)
- **Total Epochs Run:** 10

## Peak Performance Table
| Metric | Value |
| --- | --- |
| Best Epoch | 10 |
| Best Validation Accuracy | 0.9756 |
| Best Validation Loss | 0.0642 |
| Best Validation AUC | 0.9974 |

## Convergence Assessment
- **Validation loss trend:** Decreasing
- **Training loss trend:** Decreasing
- **Status:** No clear signs of immediate overfitting.


## Learning Curves
### Loss Curve
![Loss Curve](loss_curve.png)

### Accuracy & AUC Curve
![Accuracy & AUC Curve](accuracy_auc_curve.png)
