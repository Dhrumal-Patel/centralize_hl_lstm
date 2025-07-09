
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

# Dataset Class
class SensorDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create sequences with specified length and overlap
def create_sequences(data, targets, seq_length, overlap):
    step_size = max(1, int(seq_length * (1 - overlap)))
    sequences = []
    seq_targets = []
    
    for i in range(0, len(data) - seq_length + 1, step_size):
        sequences.append(data[i:i + seq_length])
        seq_targets.append(targets[i + seq_length - 1])
    
    return np.array(sequences), np.array(seq_targets)

# Upsample fall scenarios per subject/trial
def upsample_fall_scenarios(df, feature_columns, sampling_strategy=0.5, random_state=42):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    group_cols = ['subject', 'trial'] if 'trial' in df.columns else ['subject']
    upsampled_dfs = []
    
    for name, group in df.groupby(group_cols):
        print(f"Upsampling group {name}: {len(group)} samples")
        if len(group['class_encoded'].unique()) < 2:
            print(f"Group {name} has only one class. Skipping upsampling.")
            upsampled_dfs.append(group)
            continue
        
        features = group[feature_columns].values
        class_labels = group['class_encoded'].values
        scenario_labels = group['scenario_encoded'].values
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"Warning: NaNs/infinities in group {name}. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            features_resampled, class_resampled = smote.fit_resample(features, class_labels)
            print(f"Group {name} after upsampling: {len(features_resampled)} samples")
            scenario_resampled = np.zeros(len(features_resampled), dtype=np.int64)
            scenario_resampled[:len(features)] = scenario_labels
            
            if len(features_resampled) > len(features):
                synthetic_features = features_resampled[len(features):]
                synthetic_classes = class_resampled[len(features):]
                
                fall_indices = np.where(class_labels == 0)[0]
                if len(fall_indices) == 0:
                    print(f"Warning: No original fall samples in group {name}. Using all samples for NN.")
                    nn_features = features
                    nn_scenarios = scenario_labels
                else:
                    nn_features = features[fall_indices]
                    nn_scenarios = scenario_labels[fall_indices]
                
                nn = NearestNeighbors(n_neighbors=1).fit(nn_features)
                distances, indices = nn.kneighbors(synthetic_features)
                
                for i, (synth_class, nn_idx) in enumerate(zip(synthetic_classes, indices.flatten())):
                    if synth_class == 0:
                        scenario_resampled[len(features) + i] = nn_scenarios[nn_idx]
                    else:
                        nn_all = NearestNeighbors(n_neighbors=1).fit(features)
                        _, idx_all = nn_all.kneighbors(synthetic_features[i:i+1])
                        scenario_resampled[len(features) + i] = scenario_labels[idx_all.flatten()[0]]
            
            group_resampled = pd.DataFrame(
                features_resampled, columns=feature_columns
            )
            group_resampled['class_encoded'] = class_resampled
            group_resampled['scenario_encoded'] = scenario_resampled
            for col in group_cols:
                group_resampled[col] = group[col].iloc[0]
            
            valid_scenarios = set(range(16))
            group_scenarios = set(group_resampled['scenario_encoded'].unique())
            if not group_scenarios.issubset(valid_scenarios):
                raise ValueError(f"Invalid scenario_encoded in group {name}: {group_scenarios}")
            upsampled_dfs.append(group_resampled)
        except Exception as e:
            print(f"Upsampling failed for group {name}: {e}. Using original group.")
            upsampled_dfs.append(group)
    
    return pd.concat(upsampled_dfs, ignore_index=True)

# Data Loading and Preparation
def load_and_prepare_data(df, config, feature_min=None, feature_max=None):
    feature_columns = config.FEATURE_COLUMNS
    available_columns = [col for col in feature_columns]

    if not available_columns:
        raise ValueError("No valid feature columns found in the dataframe")
    
    df = df.copy()
    
    print("Upsampling fall scenarios...")
    df = upsample_fall_scenarios(df, available_columns, config.SAMPLING_STRATEGY)
    
    for col in available_columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Non-numeric or missing values found in column {col}")
    
    group_cols = ['subject', 'trial'] if 'trial' in df.columns else ['subject']
    grouped = df.groupby(group_cols)
    features = []
    targets = []
    
    for name, group in grouped:
        print(f"Processing group {name}: {len(group)} samples")
        group_features = group[available_columns].values
        group_class = group['class_encoded'].values
        group_scenario = group['scenario_encoded'].values
        
        if len(group) >= config.SEQUENCE_LENGTH:
            group_seqs, group_seq_targets = create_sequences(
                group_features, 
                np.column_stack((group_class, group_scenario)),
                config.SEQUENCE_LENGTH, 
                config.OVERLAP
            )
            if len(group_seqs) > 0:
                print(f"Group {name} produced {len(group_seqs)} sequences")
                features.append(group_seqs)
                targets.append(group_seq_targets)
        else:
            print(f"Skipping group {name}: too few samples ({len(group)} < {config.SEQUENCE_LENGTH})")
    
    if not features:
        raise ValueError("No sequences created from the data")
    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    print(f"Number of samples: {features.shape[0]}")
    unique, counts = np.unique(targets[:, 0], return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    if feature_min is not None and feature_max is not None:
        feature_min = feature_min.reshape(1, 1, -1)
        feature_max = feature_max.reshape(1, 1, -1)
        features = (features - feature_min) / (feature_max - feature_min + 1e-10)
    
    fall_map = {scenario: idx for idx, scenario in enumerate(config.FALL_SCENARIOS)}
    non_fall_map = {scenario: idx for idx, scenario in enumerate(config.NON_FALL_SCENARIOS)}
    
    remapped_scenarios = np.zeros(len(targets), dtype=np.int64)
    for scenario in fall_map:
        remapped_scenarios[targets[:, 1] == scenario] = fall_map[scenario]
    for scenario in non_fall_map:
        remapped_scenarios[targets[:, 1] == scenario] = non_fall_map[scenario]
    
    unique_scenarios = np.unique(remapped_scenarios)
    print(f"Remapped scenario values: {unique_scenarios}")
    if np.any(remapped_scenarios >= len(config.FALL_SCENARIOS + config.NON_FALL_SCENARIOS)):
        raise ValueError(f"Invalid scenario values found: {unique_scenarios}")
    
    targets[:, 1] = remapped_scenarios
    
    unique_binary = np.unique(targets[:, 0])
    unique_scenarios = np.unique(targets[:, 1])
    print(f"Final binary targets: {unique_binary}")
    print(f"Final scenario targets: {unique_scenarios}")
    
    return torch.FloatTensor(features), torch.LongTensor(targets)

# Prepare centralized data
def prepare_centralized_data(config):
    df = pd.read_csv(config.DATA_FILE)
    excluded_subjects = [41, 24, 50]
    df = df[~df['subject'].isin(excluded_subjects)]
    
    all_subjects = df['subject'].unique()
    print(f"Total subjects: {len(all_subjects)}")
    
    train_subjects, temp_subjects = train_test_split(
        all_subjects, train_size=0.70, random_state=42
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects, test_size=0.5, random_state=42
    )
    print(f"Training subjects: {len(train_subjects)} ({train_subjects})")
    print(f"Validation subjects: {len(val_subjects)} ({val_subjects})")
    print(f"Testing subjects: {len(test_subjects)} ({test_subjects})")
    
    train_df = df[df['subject'].isin(train_subjects)]
    val_df = df[df['subject'].isin(val_subjects)]
    test_df = df[df['subject'].isin(test_subjects)]
    
    print("Computing global min-max values...")
    feature_columns = config.FEATURE_COLUMNS
    available_columns = [col for col in feature_columns if col in train_df.columns]
    train_features = train_df[available_columns].values
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    config.FEATURE_MIN = np.min(train_features, axis=0)
    config.FEATURE_MAX = np.max(train_features, axis=0)
    config.FEATURE_MAX = np.where(config.FEATURE_MAX == config.FEATURE_MIN, 
                                  config.FEATURE_MAX + 1e-10, config.FEATURE_MAX)
    
    train_features, train_targets = load_and_prepare_data(
        train_df, config, config.FEATURE_MIN, config.FEATURE_MAX
    )
    val_features, val_targets = load_and_prepare_data(
        val_df, config, config.FEATURE_MIN, config.FEATURE_MAX
    )
    test_features, test_targets = load_and_prepare_data(
        test_df, config, config.FEATURE_MIN, config.FEATURE_MAX
    )
    
    return (SensorDataset(train_features, train_targets),
            SensorDataset(val_features, val_targets),
            SensorDataset(test_features, test_targets))

# Verify Dataset
def verify_dataset(config):
    df = pd.read_csv(config.DATA_FILE)
    print("Unique scenario_encoded values:", list(df['scenario_encoded'].unique()))
    print("Unique class_encoded values:", list(df['class_encoded'].unique()))
    print("Fall scenarios in dataset:", set(df[df['class_encoded'] == 0]['scenario_encoded'].unique()))
    print("Non-fall scenarios in dataset:", set(df[df['class_encoded'] == 1]['scenario_encoded'].unique()))
    for col in config.FEATURE_COLUMNS:
        if col in df.columns:
            print(f"Column {col} sample: {df[col].iloc[0]}, type: {type(df[col].iloc[0])}")
