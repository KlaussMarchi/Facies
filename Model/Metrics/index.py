import numpy as np
import pandas as pd
import torch


# MÉTRICAS POR VOXEL DE UMA MATRIZ DE CONFUSÃO classes × classes (LINHA É A CLASSE REAL, COLUNA A PREDITA), SOMADA LOTE A LOTE
class Metrics:
    MEANS     = ('accuracy', 'mean_class_accuracy', 'mean_precision', 'mean_iou', 'frequency_weighted_iou', 'mean_f1')
    PER_CLASS = ('class_accuracy', 'precision', 'iou', 'f1')

    def __init__(self, classes, confusion=None):
        self.classes   = max(int(classes), 2)
        self.confusion = np.zeros((self.classes, self.classes), dtype=np.int64) if confusion is None else np.asarray(confusion, dtype=np.int64)

    # SOMA UM LOTE NA MATRIZ: target E pred COM AS CLASSES, numpy OU torch, DE QUALQUER FORMATO (O MESMO NOS DOIS)
    def update(self, target, pred):
        n = self.classes
        if torch.is_tensor(target):
            counts = torch.bincount((target.long() * n + pred.long()).flatten(), minlength=n * n).cpu().numpy()
        else:
            counts = np.bincount((np.asarray(target, np.int64) * n + np.asarray(pred, np.int64)).ravel(), minlength=n * n)

        self.confusion += counts.reshape(n, n)
        return self

    @property
    def total(self):
        return int(self.confusion.sum())

    def summary(self):
        tp, real, predicted = np.diag(self.confusion), self.confusion.sum(axis=1), self.confusion.sum(axis=0)
        total     = max(self.total, 1)
        accuracy  = tp / np.maximum(real, 1)
        precision = tp / np.maximum(predicted, 1)
        iou       = tp / np.maximum(real + predicted - tp, 1)
        f1        = 2 * tp / np.maximum(real + predicted, 1)
        byClass   = lambda values: {str(c): float(v) for c, v in enumerate(values)}

        return {
            'accuracy': float(tp.sum() / total), 'mean_class_accuracy': float(accuracy.mean()), 'mean_precision': float(precision.mean()),
            'mean_iou': float(iou.mean()), 'frequency_weighted_iou': float((real / total * iou).sum()), 'mean_f1': float(f1.mean()),
            'class_accuracy': byClass(accuracy), 'precision': byClass(precision), 'iou': byClass(iou), 'f1': byClass(f1),
        }

    # IoU DE CADA CLASSE E A MÉDIA, O FORMATO DO test_iou E DO block_iou NO info.json
    def iou(self):
        summary = self.summary()
        return {**summary['iou'], 'mean': summary['mean_iou']}

    # TABELA POR CLASSE COM A LINHA mean (MCA, PRECISÃO MÉDIA, mIoU E F1 MÉDIO)
    def table(self):
        summary = self.summary()
        table   = pd.DataFrame({key: summary[key] for key in self.PER_CLASS})
        table.loc['mean'] = table.mean()
        return table

    def __str__(self):
        summary = self.summary()
        return ' | '.join(f'{key} {summary[key]:.4f}' for key in self.MEANS) + f' | {self.total:,} voxels'
