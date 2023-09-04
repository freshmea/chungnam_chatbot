import torch
import torchmetrics

preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))


# acc = torchmetrics.functional.accuracy(preds, target, 'multiclass')