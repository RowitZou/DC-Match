
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Metric():
    def __init__(self):
        self.metric = load_metric('glue', 'qqp')

    def compute_metrics_f1(self, pred):
        predictions, labels = pred
        if isinstance(predictions, tuple):
            preds = predictions[0].argmax(-1)
        else:
            preds = predictions.argmax(-1)
        return self.metric.compute(predictions=preds, references=labels)

    def compute_metrics_macro_f1(self, pred):
        predictions, labels = pred
        if isinstance(predictions, tuple):
            preds = predictions[0].argmax(-1)
        else:
            preds = predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro'
        )
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'macro-f1': f1,
            'precision': precision,
            'recall': recall
        }
