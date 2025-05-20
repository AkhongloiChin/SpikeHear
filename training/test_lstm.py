import torch
import numpy as np
from lstm import LSTMClassifier 
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class SpikeTrainDataset(Dataset):
    def __init__(self, data_dir, labels_csv):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

        self.labels_map = dict(zip(self.labels_df['filename'], self.labels_df['label']))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        spike_train = np.load(filepath)  
        spike_tensor = torch.tensor(spike_train, dtype=torch.float32).T  

        filename = os.path.basename(filepath)
        label = self.labels_map.get(filename, -1)
        label = torch.tensor(label, dtype=torch.long)
        return spike_tensor, label


def collate_fn(batch):
    # Pad sequences to max length in batch
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_seqs, labels, lengths

def evaluate_lstm_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_X, batch_y, lengths in test_loader:
            batch_X = batch_X.to(device) 
            batch_y = batch_y.to(device) 

            outputs = model(batch_X) 

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_targets, all_preds)

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc * 100:.2f}%")

    return acc

# Parameters
input_size = 39
hidden_size = 128
num_layers = 2
num_classes = 10  

# Load the trained model
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load("lstm_trained.pt"))
model.eval()

def predict(model, spike_train_np):
    """
    spike_train_np: np.array of shape [n_mels, time_steps] (same format as training .npy)
    """
    spike_tensor = torch.tensor(spike_train_np, dtype=torch.float32).T.unsqueeze(0)  

    with torch.no_grad():
        output = model(spike_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    return predicted_label


spike_train_sample = np.load("../data/mfcc_split_data/test/2969_spike.npy")
predicted_label = predict(model, spike_train_sample)
print(f"Predicted label: {predicted_label}")


TRAIN_DIR = "../data/mfcc_split_data/train"
TEST_DIR = "../data/mfcc_split_data/test"
LABELS_CSV = "../data/labels.csv"

# Load datasets
train_dataset = SpikeTrainDataset(TRAIN_DIR, LABELS_CSV)
test_dataset = SpikeTrainDataset(TEST_DIR, LABELS_CSV)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

print('acc: ', evaluate_lstm_model(model, test_loader,device = torch.device('cpu')))