
import os
import torch
import csv

# Set a safe CSV field size limit
csv.field_size_limit(2147483647)

# Configuration
class Config:
    def __init__(self, overlap=0.0):
        self.OVERLAP = overlap
        self.SAVE_FOLDER = os.path.join(os.getcwd(), f'federated_results/overlap__{}'.format(overlap)
        self.DATA_FILE = '/content/drive/My Drive/FL_LSTM_CODE/Datasets/mobiact2.csv'  # Path to extracted CSV
        self.ZIP_FILE = 'E:/DA-IICT/SEM-2/Minor_Project/Final Paper/Code/FL_LSTM_CODE/Datasets/mobiact2.csv'
        self.BATCH_SIZE = 16
        self.CLIENT_EPOCHS = 10
        self.LEARNING_RATE = 0.01
        self.HIDDEN_SIZE_BINARY = 128
        self.HIDDEN_SIZE_MULTICLASS = 256
        self.NUM_LAYERS = 2
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FALL_SCENARIOS = [5, 4, 10, 0]
        self.NON_FALL_SCENARIOS = [i for i in range(16) if i not in [5, 4, 10, 0]]
        self.FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll']
        self.MAX_PARAM_SIZE = 100000
        self.SEQUENCE_LENGTH = 50
        self.SAMPLING_STRATEGY = 0.5
