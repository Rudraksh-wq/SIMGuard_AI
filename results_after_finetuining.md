# Fine-tuning results (Laya, before and after)

| Metric | Base Laya (untouched) | Fine-tuned (a1_6ep, epoch 5) |
|---|---|---|
| Validation accuracy | 0.2517 | 0.9937 |
| Validation loss | 1.735 | 0.0201 |
| Test accuracy | 0.2565 | 0.9917 |
| Hard dev set accuracy | 0.553 | 0.837 |
| Hard test set accuracy | 0.550 | 0.833 |
| Balanced accuracy | 0.50 (no signal at all) | ~0.99 |
| Speed (GPU, single case, median) | 40.9 ms | 39.1 ms |

All results are on synthetic data. Full details (per-category results, decision gate, independent
re-verification, limitations): [finetune/fraud_v2/runs/PERFORMANCE_REPORT.md](finetune/fraud_v2/runs/PERFORMANCE_REPORT.md).
The model itself: [Kyoiske/simguard-laya-fraud](https://huggingface.co/Kyoiske/simguard-laya-fraud).
