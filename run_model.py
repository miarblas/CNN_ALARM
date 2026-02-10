import os
import json
import torch
import numpy as np
import pandas as pd
import argparse

from dataset import H5AlarmDataset, PreprocessConfig
from model import SimpleCNN1D
from train_utils import get_device

def run_inference(test_h5_path: str, output_csv: str = 'predictions.csv'):
    """
    Run inference on the test set (provided separately).
    
    Args:
        test_h5_path: Path to the test H5 file
        output_csv: Path to save predictions
    """
    
    print("=" * 60)
    print("ICU FALSE ALARM DETECTION - INFERENCE")
    print("=" * 60)
    
    # Check if test file exists
    if not os.path.exists(test_h5_path):
        raise FileNotFoundError(f"Test file not found: {test_h5_path}")
    
    # Load configuration from training
    artifacts_dir = 'artifacts'
    config_path = os.path.join(artifacts_dir, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. Train model first.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load preprocessing config
    pp_dict = config['preprocess']
    pp = PreprocessConfig(
        target_fs=pp_dict['target_fs'],
        window_sec=pp_dict['window_sec'],
        max_channels=pp_dict['max_channels']
    )
    
    # Load optimal threshold
    optimal_threshold = config['training']['optimal_threshold']
    print(f"Using optimal threshold: {optimal_threshold:.4f}")
    
    # Load test dataset
    print(f"\nLoading test data from {test_h5_path}...")
    
    # Check if labels exist in test set
    import h5py
    with h5py.File(test_h5_path, 'r') as f:
        has_labels = 'status' in f
    
    if has_labels:
        print("Test set contains labels - will calculate metrics")
        test_ds = H5AlarmDataset(test_h5_path, pp=pp, require_labels=True)
    else:
        print("Test set doesn't contain labels - inference only")
        test_ds = H5AlarmDataset(test_h5_path, pp=pp, require_labels=False)
    
    print(f"Test samples: {len(test_ds)}")
    
    # Setup model
    device = get_device()
    model = SimpleCNN1D(in_channels=pp.max_channels).to(device)
    
    # Load trained weights
    model_path = os.path.join(artifacts_dir, 'model.pth')
    if not os.path.exists(model_path):
        # Try best model
        model_path = os.path.join(artifacts_dir, 'best_model.pth')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    
    # Run inference
    print(f"\nRunning inference...")
    predictions = []
    probabilities = []
    true_labels = [] if has_labels else None
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            
            logits = model(x)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= optimal_threshold else 0
            
            probabilities.append(prob)
            predictions.append(pred)
            
            if has_labels:
                true_labels.append(y.item())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_ds)} samples")
    
    print(f"Inference complete!")
    
    # Create output DataFrame
    df = pd.DataFrame({
        'sample_id': np.arange(len(predictions)),
        'probability': probabilities,
        'prediction': predictions,
        'is_true_alarm': predictions  # 1 = True alarm, 0 = False alarm
    })
    
    # If we have true labels, add them and calculate metrics
    if has_labels:
        df['true_label'] = true_labels
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        
        try:
            auc = roc_auc_score(true_labels, probabilities)
        except:
            auc = float('nan')
        
        print(f"\n{'='*60}")
        print("TEST SET METRICS")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"AUC:       {auc:.4f}")
        
        # Confusion matrix
        tp = sum((p == 1) and (t == 1) for p, t in zip(predictions, true_labels))
        tn = sum((p == 0) and (t == 0) for p, t in zip(predictions, true_labels))
        fp = sum((p == 1) and (t == 0) for p, t in zip(predictions, true_labels))
        fn = sum((p == 0) and (t == 1) for p, t in zip(predictions, true_labels))
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  False Alarm Rate: {fp/(fp+tn+1e-10):.2%}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to: {output_csv}")
    
    # Summary
    true_alarms = sum(predictions)
    false_alarms = len(predictions) - true_alarms
    
    print(f"\nPredictions Summary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Predicted True Alarms:  {true_alarms} ({true_alarms/len(predictions):.1%})")
    print(f"  Predicted False Alarms: {false_alarms} ({false_alarms/len(predictions):.1%})")
    print(f"  Mean probability: {np.mean(probabilities):.4f}")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--test_path', type=str, default='data/testSet.h5',
                       help='Path to test H5 file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file name')
    
    args = parser.parse_args()
    
    run_inference(args.test_path, args.output)