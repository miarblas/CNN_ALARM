import numpy as np
import torch
import constants as const

from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, min_delta=0, verbose=False, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_metrics = None

    def __call__(self, val_loss, epoch, model, metrics=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, metrics)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, metrics)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, metrics):
        if self.verbose:
            print(f'Validation loss improved ({self.best_score:.6f} --> {-val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_epoch = epoch
        self.best_metrics = metrics


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = 'f1') -> Tuple[float, Dict]:
    """
    Find optimal threshold for binary classification based on specified metric.
    Returns optimal threshold and associated metrics.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    results = {
        'thresholds': thresholds,
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'specificities': [],
        'accuracies': []
    }

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        accuracy = (tp + tn) / max(1, len(y_true))

        results['f1_scores'].append(f1)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['specificities'].append(specificity)
        results['accuracies'].append(accuracy)

    # Select best threshold
    if metric == 'f1':
        best_idx = np.argmax(results['f1_scores'])
    elif metric == 'f1_weighted':
        weights = [recall if y_true.mean() > 0.5 else results['specificities'][i] 
                   for i, recall in enumerate(results['recalls'])]
        weighted_f1 = [f1 * w for f1, w in zip(results['f1_scores'], weights)]
        best_idx = np.argmax(weighted_f1)
    elif metric == 'youden':
        youden_scores = [r + s - 1 for r, s in zip(results['recalls'], results['specificities'])]
        best_idx = np.argmax(youden_scores)
    elif metric == 'balanced_acc':
        balanced_acc = [(r + s) / 2 for r, s in zip(results['recalls'], results['specificities'])]
        best_idx = np.argmax(balanced_acc)
    else:
        best_idx = np.argmax(results['f1_scores'])

    optimal_threshold = thresholds[best_idx]
    best_results = {
        'threshold': optimal_threshold,
        'f1': results['f1_scores'][best_idx],
        'precision': results['precisions'][best_idx],
        'recall': results['recalls'][best_idx],
        'specificity': results['specificities'][best_idx],
        'accuracy': results['accuracies'][best_idx]
    }
    return optimal_threshold, best_results


@torch.no_grad()
def get_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Get true labels and predicted probabilities from model."""
    model.eval()
    all_probs, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: Optional[float] = None) -> Dict[str, float]:
    """Evaluate model and compute metrics, including confusion matrix and custom score."""
    threshold = const.THRESHOLD if threshold is None else threshold
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    all_probs, all_labels = [], []
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    # Concatenate
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(int)

    # Confusion matrix
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    # Metrics
    eps = 1e-10
    acc = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(eps, precision + recall)
    balanced_acc = (recall + specificity) / 2
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5
    false_alarm_rate = fp / max(1, fp + tn)

    # Custom score: TP+TN / (TP+TN+FP+5*FN)
    custom_score = (tp + tn) / max(1, tp + tn + fp + 5 * fn)

    return {
        "loss": total_loss / max(1, len(labels)),
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "auc": auc,
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_pct": 100 * false_alarm_rate,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "custom_score": custom_score
    }


def get_device() -> torch.device:
    """Get available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
