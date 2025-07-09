
import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from config import Config
from dataset import prepare_centralized_data, verify_dataset
from model import LSTMModel
import zipfile

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_keys_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_keys_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_json_serializable(item) for item in obj]
    return obj

def extract_zip(zip_path, extract_path):
    """Extract the ZIP file to the specified path."""
    print(f"Extracting {zip_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction complete. Files extracted to {extract_path}")

def compute_model_norm(model):
    """Compute the L2 norm of model parameters for debugging."""
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param, p=2).item() ** 2
    return np.sqrt(norm)

def evaluate_model(models, loader, config, model_name='test'):
    """Evaluate the models on the given dataset."""
    for model_name in models:
        models[model_name].eval()
    
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'three_models': {
            'binary_correct': 0,
            'fall_correct': 0,
            'non_fall_correct': 0,
            'total': 0,
            'fall_total': 0,
            'non_fall_total': 0,
            'binary_preds': [],
            'binary_targets': [],
            'fall_preds': [],
            'fall_targets': [],
            'non_fall_preds': [],
            'non_fall_targets': [],
            'binary_class_counts': {},
            'fall_class_counts': {},
            'non_fall_class_counts': {}
        },
        'two_models': {
            'binary_acc': 0.0,
            'binary_weighted_acc': 0.0,
            'binary_precision': 0.0,
            'binary_recall': 0.0,
            'binary_f1': 0.0,
            'multiclass_acc': 0.0,
            'multiclass_weighted_acc': 0.0,
            'multiclass_precision': 0.0,
            'multiclass_recall': 0.0,
            'multiclass_f1': 0.0
        }
    }
    
    multiclass_preds = []
    multiclass_targets = []
    multiclass_class_counts = {}
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            binary_targets, multi_targets = targets[:, 0], targets[:, 1]
            
            binary_out = models['binary'](inputs).to(config.DEVICE)
            _, binary_preds = torch.max(binary_out, 1)
            metrics['three_models']['binary_correct'] += (binary_preds == binary_targets).sum().item()
            metrics['three_models']['binary_preds'].extend(binary_preds.tolist())
            metrics['three_models']['binary_targets'].extend(binary_targets.tolist())
            
            for target in binary_targets.cpu().numpy():
                metrics['three_models']['binary_class_counts'][int(target)] = metrics['three_models']['binary_class_counts'].get(int(target), 0) + 1
            
            fall_mask = binary_targets == 0
            if fall_mask.any():
                fall_out = models['fall'](inputs[fall_mask]).to(config.DEVICE)
                _, fall_preds = torch.max(fall_out, 1)
                metrics['three_models']['fall_correct'] += (fall_preds == multi_targets[fall_mask]).sum().item()
                metrics['three_models']['fall_total'] += fall_mask.sum().item()
                metrics['three_models']['fall_preds'].extend(fall_preds.tolist())
                metrics['three_models']['fall_targets'].extend(multi_targets[fall_mask].tolist())
                for target in multi_targets[fall_mask].cpu().numpy():
                    metrics['three_models']['fall_class_counts'][int(target)] = metrics['three_models']['fall_class_counts'].get(int(target), 0) + 1
                
                for pred, target in zip(fall_preds.tolist(), multi_targets[fall_mask].tolist()):
                    multiclass_preds.append(pred)
                    multiclass_targets.append(target)
                    multiclass_class_counts[target] = multiclass_class_counts.get(target, 0) + 1
            
            non_fall_mask = binary_targets == 1
            if non_fall_mask.any():
                non_fall_out = models['non_fall'](inputs[non_fall_mask]).to(config.DEVICE)
                _, non_fall_preds = torch.max(non_fall_out, 1)
                metrics['three_models']['non_fall_correct'] += (non_fall_preds == multi_targets[non_fall_mask]).sum().item()
                metrics['three_models']['non_fall_total'] += non_fall_mask.sum().item()
                metrics['three_models']['non_fall_preds'].extend(non_fall_preds.tolist())
                metrics['three_models']['non_fall_targets'].extend(multi_targets[non_fall_mask].tolist())
                for target in multi_targets[non_fall_mask].cpu().numpy():
                    metrics['three_models']['non_fall_class_counts'][int(target)] = metrics['three_models']['non_fall_class_counts'].get(int(target), 0) + 1
                
                for pred, target in zip(non_fall_preds.tolist(), multi_targets[non_fall_mask].tolist()):
                    multiclass_preds.append(pred)
                    multiclass_targets.append(target)
                    multiclass_class_counts[target] = multiclass_class_counts.get(target, 0) + 1
            
            metrics['three_models']['total'] += len(targets)
    
    if metrics['three_models']['binary_class_counts']:
        binary_weights = np.array([metrics['three_models']['binary_class_counts'].get(i, 0) for i in range(2)])
        binary_weights = binary_weights / binary_weights.sum() if binary_weights.sum() > 0 else np.ones(2) / 2
        binary_correct_weighted = 0
        for i in range(2):
            correct_count = sum(1 for pred, target in zip(metrics['three_models']['binary_preds'], metrics['three_models']['binary_targets']) if pred == target == i)
            binary_correct_weighted += correct_count * binary_weights[i]
        metrics['three_models']['binary_weighted_acc'] = binary_correct_weighted / max(1, metrics['three_models']['total'])
    
    if metrics['three_models']['fall_class_counts']:
        fall_weights = np.array([metrics['three_models']['fall_class_counts'].get(i, 0) for i in range(len(config.FALL_SCENARIOS))])
        fall_weights = fall_weights / fall_weights.sum() if fall_weights.sum() > 0 else np.ones(len(config.FALL_SCENARIOS)) / len(config.FALL_SCENARIOS)
        fall_correct_weighted = 0
        for i in range(len(config.FALL_SCENARIOS)):
            correct_count = sum(1 for pred, target in zip(metrics['three_models']['fall_preds'], metrics['three_models']['fall_targets']) if pred == target == i)
            fall_correct_weighted += correct_count * fall_weights[i]
        metrics['three_models']['fall_weighted_acc'] = fall_correct_weighted / max(1, metrics['three_models']['fall_total'])
    
    if metrics['three_models']['non_fall_class_counts']:
        non_fall_weights = np.array([metrics['three_models']['non_fall_class_counts'].get(i, 0) for i in range(len(config.NON_FALL_SCENARIOS))])
        non_fall_weights = non_fall_weights / non_fall_weights.sum() if non_fall_weights.sum() > 0 else np.ones(len(config.NON_FALL_SCENARIOS)) / len(config.NON_FALL_SCENARIOS)
        non_fall_correct_weighted = 0
        for i in range(len(config.NON_FALL_SCENARIOS)):
            correct_count = sum(1 for pred, target in zip(metrics['three_models']['non_fall_preds'], metrics['three_models']['non_fall_targets']) if pred == target == i)
            non_fall_correct_weighted += correct_count * non_fall_weights[i]
        metrics['three_models']['non_fall_weighted_acc'] = non_fall_correct_weighted / max(1, metrics['three_models']['non_fall_total'])
    
    metrics['three_models']['binary_acc'] = metrics['three_models']['binary_correct'] / max(1, metrics['three_models']['total'])
    metrics['three_models']['fall_acc'] = metrics['three_models']['fall_correct'] / max(1, metrics['three_models']['fall_total'])
    metrics['three_models']['non_fall_acc'] = metrics['three_models']['non_fall_correct'] / max(1, metrics['three_models']['non_fall_total'])
    
    if metrics['three_models']['binary_targets']:
        binary_prf = precision_recall_fscore_support(
            metrics['three_models']['binary_targets'], metrics['three_models']['binary_preds'], average='weighted', zero_division=0)
        metrics['three_models']['binary_precision'] = binary_prf[0]
        metrics['three_models']['binary_recall'] = binary_prf[1]
        metrics['three_models']['binary_f1'] = binary_prf[2]
    
    if metrics['three_models']['fall_targets']:
        fall_prf = precision_recall_fscore_support(
            metrics['three_models']['fall_targets'], metrics['three_models']['fall_preds'], average='weighted', zero_division=0)
        metrics['three_models']['fall_precision'] = fall_prf[0]
        metrics['three_models']['fall_recall'] = fall_prf[1]
        metrics['three_models']['fall_f1'] = fall_prf[2]
    
    if metrics['three_models']['non_fall_targets']:
        non_fall_prf = precision_recall_fscore_support(
            metrics['three_models']['non_fall_targets'], metrics['three_models']['non_fall_preds'], average='weighted', zero_division=0)
        metrics['three_models']['non_fall_precision'] = non_fall_prf[0]
        metrics['three_models']['non_fall_recall'] = non_fall_prf[1]
        metrics['three_models']['non_fall_f1'] = non_fall_prf[2]
    
    metrics['two_models']['binary_acc'] = metrics['three_models']['binary_acc']
    metrics['two_models']['binary_weighted_acc'] = metrics['three_models']['binary_weighted_acc']
    metrics['two_models']['binary_precision'] = metrics['three_models']['binary_precision']
    metrics['two_models']['binary_recall'] = metrics['three_models']['binary_recall']
    metrics['two_models']['binary_f1'] = metrics['three_models']['binary_f1']
    
    total_multiclass_samples = metrics['three_models']['fall_total'] + metrics['three_models']['non_fall_total']
    if total_multiclass_samples > 0:
        multiclass_correct = metrics['three_models']['fall_correct'] + metrics['three_models']['non_fall_correct']
        metrics['two_models']['multiclass_acc'] = multiclass_correct / total_multiclass_samples
        
        if multiclass_class_counts:
            num_classes = len(config.FALL_SCENARIOS) + len(config.NON_FALL_SCENARIOS)
            multiclass_weights = np.array([multiclass_class_counts.get(i, 0) for i in range(num_classes)])
            multiclass_weights = multiclass_weights / multiclass_weights.sum() if multiclass_weights.sum() > 0 else np.ones(num_classes) / num_classes
            multiclass_correct_weighted = 0
            for i in range(num_classes):
                correct_count = sum(1 for pred, target in zip(multiclass_preds, multiclass_targets) if pred == target == i)
                multiclass_correct_weighted += correct_count * multiclass_weights[i]
            metrics['two_models']['multiclass_weighted_acc'] = multiclass_correct_weighted / max(1, total_multiclass_samples)
        
        if multiclass_targets:
            multiclass_prf = precision_recall_fscore_support(
                multiclass_targets, multiclass_preds, average='weighted', zero_division=0)
            metrics['two_models']['multiclass_precision'] = multiclass_prf[0]
            metrics['two_models']['multiclass_recall'] = multiclass_prf[1]
            metrics['two_models']['multiclass_f1'] = multiclass_prf[2]
    
    return convert_keys_to_json_serializable(metrics)

def train_models(models, train_loader, config, epochs):
    """Train all models on the centralized dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        'binary': torch.optim.Adam(models['binary'].parameters(), lr=config.LEARNING_RATE),
        'fall': torch.optim.Adam(models['fall'].parameters(), lr=config.LEARNING_RATE),
        'non_fall': torch.optim.Adam(models['non_fall'].parameters(), lr=config.LEARNING_RATE)
    }
    
    for epoch in range(epochs):
        print(f"Training epoch {epoch+1}/{epochs}")
        for model_name in models:
            models[model_name].train()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                inputs, targets = batch
                inputs = inputs.to(config.DEVICE)
                binary_targets, scenario_targets = targets[:, 0].long().to(config.DEVICE), targets[:, 1].long().to(config.DEVICE)
                
                optimizers['binary'].zero_grad()
                binary_out = models['binary'](inputs)
                binary_loss = criterion(binary_out, binary_targets)
                binary_loss.backward()
                optimizers['binary'].step()
                
                fall_mask = binary_targets == 0
                if fall_mask.any():
                    optimizers['fall'].zero_grad()
                    fall_out = models['fall'](inputs[fall_mask])
                    fall_loss = criterion(fall_out, scenario_targets[fall_mask])
                    fall_loss.backward()
                    optimizers['fall'].step()
                
                non_fall_mask = binary_targets == 1
                if non_fall_mask.any():
                    optimizers['non_fall'].zero_grad()
                    non_fall_out = models['non_fall'](inputs[non_fall_mask])
                    non_fall_loss = criterion(non_fall_out, scenario_targets[non_fall_mask])
                    non_fall_loss.backward()
                    optimizers['non_fall'].step()
            
            except Exception as e:
                print(f"Error processing batch {batch_idx+1}: {str(e)}")
                continue
        
        val_metrics = evaluate_model(models, val_loader, config, model_name='validation')
        print(f"\nValidation Metrics (Three Models) - Epoch {epoch+1}")
        print(f"Binary: Acc={val_metrics['three_models']['binary_acc']:.4f}, "
              f"W-Acc={val_metrics['three_models']['binary_weighted_acc']:.4f}, "
              f"Prec={val_metrics['three_models']['binary_precision']:.4f}, "
              f"Rec={val_metrics['three_models']['binary_recall']:.4f}, "
              f"F1={val_metrics['three_models']['binary_f1']:.4f}")
        print(f"Fall: Acc={val_metrics['three_models']['fall_acc']:.4f}, "
              f"W-Acc={val_metrics['three_models']['fall_weighted_acc']:.4f}, "
              f"Prec={val_metrics['three_models']['fall_precision']:.4f}, "
              f"Rec={val_metrics['three_models']['fall_recall']:.4f}, "
              f"F1={val_metrics['three_models']['fall_f1']:.4f}")
        print(f"Non-Fall: Acc={val_metrics['three_models']['non_fall_acc']:.4f}, "
              f"W-Acc={val_metrics['three_models']['non_fall_weighted_acc']:.4f}, "
              f"Prec={val_metrics['three_models']['non_fall_precision']:.4f}, "
              f"Rec={val_metrics['three_models']['non_fall_recall']:.4f}, "
              f"F1={val_metrics['three_models']['non_fall_f1']:.4f}")
        print(f"Validation Metrics (Two Models) - "
              f"Binary: Acc={val_metrics['two_models']['binary_acc']:.4f}, "
              f"W-Acc={val_metrics['two_models']['binary_weighted_acc']:.4f}, "
              f"Prec={val_metrics['two_models']['binary_precision']:.4f}, "
              f"Rec={val_metrics['two_models']['binary_recall']:.4f}, "
              f"F1={val_metrics['two_models']['binary_f1']:.4f}, "
              f"Multiclass: Acc={val_metrics['two_models']['multiclass_acc']:.4f}, "
              f"W-Acc={val_metrics['two_models']['multiclass_weighted_acc']:.4f}, "
              f"Prec={val_metrics['two_models']['multiclass_precision']:.4f}, "
              f"Rec={val_metrics['two_models']['multiclass_recall']:.4f}, "
              f"F1={val_metrics['two_models']['multiclass_f1']:.4f}")
    
    return models

def save_model_to_csv(model, file_path, config):
    """Save model parameters and metadata to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    state_dict = model.state_dict()
    
    metadata = {
        'input_size': 9,
        'hidden_size': config.HIDDEN_SIZE_BINARY if 'binary' in file_path else config.HIDDEN_SIZE_MULTICLASS,
        'num_layers': config.NUM_LAYERS,
        'num_classes': (2 if 'binary' in file_path else
                        len(config.FALL_SCENARIOS) if 'fall' in file_path else
                        len(config.NON_FALL_SCENARIOS))
    }
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['__metadata__'] + [f"{k}={v}" for k, v in metadata.items()])
        for name, param in state_dict.items():
            param_flat = param.cpu().numpy().flatten()
            writer.writerow([name] + param_flat.tolist())

if __name__ == "__main__":
    overlap_values = [0.0]
    for overlap in overlap_values:
        print(f"\n=== Running Centralized Training with Overlap = {overlap} ===")
        config = Config(overlap=overlap)
        os.makedirs(config.SAVE_FOLDER, exist_ok=True)
        
        # Extract ZIP file if CSV doesn't exist
        if not os.path.exists(config.DATA_FILE):
            if os.path.exists(config.ZIP_FILE):
                extract_zip(config.ZIP_FILE, os.path.dirname(config.DATA_FILE))
            else:
                raise FileNotFoundError(f"ZIP file not found at {config.ZIP_FILE}")
        
        # Verify dataset
        verify_dataset(config)
        
        # Prepare data
        try:
            train_dataset, val_dataset, test_dataset = prepare_centralized_data(config)
            print(f"Prepared centralized train, validation, and test datasets.")
        except Exception as e:
            print(f"Error preparing datasets: {e}")
            continue
        
        # Initialize models
        models = {
            'binary': LSTMModel(9, config.HIDDEN_SIZE_BINARY, config.NUM_LAYERS, 2).to(config.DEVICE),
            'fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.FALL_SCENARIOS)).to(config.DEVICE),
            'non_fall': LSTMModel(9, config.HIDDEN_SIZE_MULTICLASS, config.NUM_LAYERS, len(config.NON_FALL_SCENARIOS)).to(config.DEVICE)
        }
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Train models
        models = train_models(models, train_loader, config, config.CLIENT_EPOCHS)
        
        # Save models
        for model_name in models:
            save_model_to_csv(
                models[model_name],
                os.path.join(config.SAVE_FOLDER, f'{model_name}_params.csv'),
                config
            )
        
        # Final evaluation on test set
        test_metrics = evaluate_model(models, test_loader, config, model_name='test')
        print(f"\nTest Metrics (Three Models) - "
              f"Binary: Acc={test_metrics['three_models']['binary_acc']:.4f}, "
              f"W-Acc={test_metrics['three_models']['binary_weighted_acc']:.4f}, "
              f"Prec={test_metrics['three_models']['binary_precision']:.4f}, "
              f"Rec={test_metrics['three_models']['binary_recall']:.4f}, "
              f"F1={test_metrics['three_models']['binary_f1']:.4f}, "
              f"Fall: Acc={test_metrics['three_models']['fall_acc']:.4f}, "
              f"W-Acc={test_metrics['three_models']['fall_weighted_acc']:.4f}, "
              f"Prec={test_metrics['three_models']['fall_precision']:.4f}, "
              f"Rec={test_metrics['three_models']['fall_recall']:.4f}, "
              f"F1={test_metrics['three_models']['fall_f1']:.4f}, "
              f"Non-Fall: Acc={test_metrics['three_models']['non_fall_acc']:.4f}, "
              f"W-Acc={test_metrics['three_models']['non_fall_weighted_acc']:.4f}, "
              f"Prec={test_metrics['three_models']['non_fall_precision']:.4f}, "
              f"Rec={test_metrics['three_models']['non_fall_recall']:.4f}, "
              f"F1={test_metrics['three_models']['non_fall_f1']:.4f}")
        print(f"Test Metrics (Two Models) - "
              f"Binary: Acc={test_metrics['two_models']['binary_acc']:.4f}, "
              f"W-Acc={test_metrics['two_models']['binary_weighted_acc']:.4f}, "
              f"Prec={test_metrics['two_models']['binary_precision']:.4f}, "
              f"Rec={test_metrics['two_models']['binary_recall']:.4f}, "
              f"F1={test_metrics['two_models']['binary_f1']:.4f}, "
              f"Multiclass: Acc={test_metrics['two_models']['multiclass_acc']:.4f}, "
              f"W-Acc={test_metrics['two_models']['multiclass_weighted_acc']:.4f}, "
              f"Prec={test_metrics['two_models']['multiclass_precision']:.4f}, "
              f"Rec={test_metrics['two_models']['multiclass_recall']:.4f}, "
              f"F1={test_metrics['two_models']['multiclass_f1']:.4f}")
        
        # Save results
        results = {
            'test_metrics': test_metrics
        }
        results = convert_keys_to_json_serializable(results)
        with open(os.path.join(config.SAVE_FOLDER, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        
        with open(os.path.join(config.SAVE_FOLDER, 'results.csv'), 'w', newline='') as f:
            metric_keys = [
                'binary_acc', 'binary_weighted_acc', 'binary_precision', 'binary_recall', 'binary_f1',
                'fall_acc', 'fall_weighted_acc', 'fall_precision', 'fall_recall', 'fall_f1',
                'non_fall_acc', 'non_fall_weighted_acc', 'non_fall_precision', 'non_fall_recall', 'non_fall_f1'
            ]
            writer = csv.DictWriter(f, fieldnames=['num_samples'] + metric_keys)
            writer.writeheader()
            row = {'num_samples': len(train_dataset)}
            row.update({k: test_metrics['three_models'].get(k, 0.0) for k in metric_keys})
            writer.writerow(row)
        print(f"Results saved to {os.path.join(config.SAVE_FOLDER, 'results.csv')}")