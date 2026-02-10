import os
import json
import torch
import numpy as np
from datetime import datetime

from dataclasses import asdict
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import constants as const
from dataset import H5AlarmDataset, PreprocessConfig
from model import SimpleCNN1D
from train_utils import (
    evaluate, get_device, EarlyStopping, 
    get_predictions, find_optimal_threshold
)

def train_model():
    """Train model on the provided trainSet.h5 only."""

    print("=" * 60)
    print("ICU FALSE ALARM DETECTION - TRAINING")
    print("=" * 60)
    print("NOTE: Using entire provided trainSet.h5")
    print("Test set will be evaluated separately\n")

    # Set random seeds for reproducibility
    torch.manual_seed(const.RANDOM_SEED)
    np.random.seed(const.RANDOM_SEED)

    # Load training dataset
    data_path = os.path.join('data', 'trainSet.h5')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Missing {data_path}. Put the provided H5 file into the data/ folder.')

    pp = PreprocessConfig(
        window_sec=const.WINDOW_SIZE,
        max_channels=const.MAX_CHANNELS
    )

    print(f"Loading training dataset from {data_path}...")
    dataset = H5AlarmDataset(data_path, pp=pp, require_labels=True)
    print(f"Total samples in trainSet.h5: {len(dataset)}")

    # Prepare stratified train/validation split
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=const.VAL_FRACTION,
        random_state=const.RANDOM_SEED,
        stratify=labels
    )

    print(f"\nDataset split (train/val):")
    print(f"  Training samples:   {len(train_idx):>6} ({len(train_idx)/len(dataset):.1%})")
    print(f"  Validation samples: {len(val_idx):>6} ({len(val_idx)/len(dataset):.1%})")

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    print(f"\nClass distribution in training set:")
    print(f"  True alarms (1):  {train_labels.sum():>6} ({train_labels.mean():.1%})")
    print(f"  False alarms (0): {(len(train_labels) - train_labels.sum()):>6} ({1-train_labels.mean():.1%})")

    print(f"\nClass distribution in validation set:")
    print(f"  True alarms (1):  {val_labels.sum():>6} ({val_labels.mean():.1%})")
    print(f"  False alarms (0): {(len(val_labels) - val_labels.sum()):>6} ({1-val_labels.mean():.1%})")

    # Datasets and loaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=const.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=const.BATCH_SIZE*2, shuffle=False, num_workers=0)

    # Model setup
    print(f"\n{'='*60}")
    print("MODEL SETUP")
    print(f"{'='*60}")

    device = get_device()
    model = SimpleCNN1D(in_channels=pp.max_channels).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SimpleCNN1D")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=const.LEARNING_RATE,
        weight_decay=const.WEIGHT_DECAY
    )

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(const.POS_WEIGHT).to(device)
    )

    # Early stopping
    artifacts_dir = const.ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)
    best_model_path = os.path.join(artifacts_dir, const.BEST_MODEL_WEIGHTS)

    early_stopping = EarlyStopping(
        patience=const.PATIENCE,
        min_delta=0.001,
        verbose=False,
        checkpoint_path=best_model_path
    )

    # TRAINING LOOP
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    best_val_f1 = 0

    for epoch in range(1, const.EPOCHS + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation phase (standard metrics only)
        val_metrics = evaluate(model, val_loader, device, threshold=const.DEFAULT_THRESHOLD)
        avg_val_loss = val_metrics['loss']

        if (epoch % 5 == 0) or (epoch == 1) or (epoch == const.EPOCHS):
            print(f"Epoch {epoch:3d}/{const.EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            print(f"✓ New best model at epoch {epoch}: Val F1={val_metrics['f1']:.4f}, Val AUC={val_metrics['auc']:.4f}")

        early_stopping(avg_val_loss, epoch, model, val_metrics)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            break

    # LOAD BEST MODEL
    print(f"\nLoading best model from epoch {early_stopping.best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # THRESHOLD OPTIMIZATION (no plot)
    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")

    y_true_val, y_prob_val = get_predictions(model, val_loader, device)
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_true_val, y_prob_val, metric=const.THRESHOLD_TUNING_METRIC
    )

    print(f"\nOptimal threshold found: {optimal_threshold:.4f}")
    print(f"  F1 Score:      {threshold_metrics['f1']:.4f}")
    print(f"  Precision:     {threshold_metrics['precision']:.4f}")
    print(f"  Recall:        {threshold_metrics['recall']:.4f}")
    print(f"  Specificity:   {threshold_metrics['specificity']:.4f}")

    # FINAL VALIDATION (custom score included)
    final_val_metrics = evaluate(model, val_loader, device, threshold=optimal_threshold)

    print(f"\n{'='*60}")
    print("FINAL VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy:  {final_val_metrics['accuracy']:.4f}")
    print(f"F1 Score:  {final_val_metrics['f1']:.4f}")
    print(f"Precision: {final_val_metrics['precision']:.4f}")
    print(f"Recall:    {final_val_metrics['recall']:.4f}")
    print(f"AUC:       {final_val_metrics['auc']:.4f}")
    print(f"False Alarm Rate: {final_val_metrics['false_alarm_pct']:.2f}%")
    print(f"Custom Score: {final_val_metrics['custom_score']:.4f}")
    print(f"Confusion Matrix: {final_val_metrics['confusion_matrix']}")

    # Save final model and config
    final_model_path = os.path.join(artifacts_dir, const.MODEL_WEIGHTS)
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    config = {
        'preprocess': asdict(pp),
        'training': {
            'best_epoch': early_stopping.best_epoch,
            'total_epochs_trained': epoch,
            'optimal_threshold': float(optimal_threshold),
            'threshold_tuning_metric': const.THRESHOLD_TUNING_METRIC,
            'final_val_metrics': final_val_metrics,
            'random_seed': const.RANDOM_SEED,
            'timestamp': datetime.now().isoformat()
        },
        'model': {
            'architecture': 'SimpleCNN1D',
            'in_channels': pp.max_channels,
            'base_filters': const.MODEL_BASE,
            'dropout': const.MODEL_DROPOUT
        }
    }

    config_path = os.path.join(artifacts_dir, const.CONFIG_FILE)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"✓ Configuration saved to: {config_path}")

    summary = {
        'dataset_info': {
            'total_samples': len(dataset),
            'training_samples': len(train_idx),
            'validation_samples': len(val_idx),
            'train_true_alarms': int(train_labels.sum()),
            'train_false_alarms': int(len(train_labels) - train_labels.sum()),
            'val_true_alarms': int(val_labels.sum()),
            'val_false_alarms': int(len(val_labels) - val_labels.sum())
        },
        'training_results': {
            'best_epoch': early_stopping.best_epoch,
            'best_val_f1': best_val_f1,
            'optimal_threshold': float(optimal_threshold),
            'final_val_metrics': final_val_metrics
        }
    }

    summary_path = os.path.join(artifacts_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary saved to: {summary_path}")

    print(f"\n{'='*60}")
    print("🎯 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model is ready for the test set.")
    print(f"To run inference on test set, use: python run_model.py")

    return model, optimal_threshold, final_val_metrics

if __name__ == '__main__':
    train_model()

