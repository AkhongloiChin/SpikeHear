import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpikeTrainDataset(Dataset):
    def __init__(self, data_dir, labels_csv):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Map filename -> label
        self.labels_map = dict(zip(self.labels_df['filename'], self.labels_df['label']))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        spike_train = np.load(filepath)  # shape: [n_mels, time_steps]
        spike_tensor = torch.tensor(spike_train, dtype=torch.float32).T  # transpose -> [time_steps, n_mels]

        filename = os.path.basename(filepath)
        label = self.labels_map.get(filename, -1)
        label = torch.tensor(label, dtype=torch.long)
        return spike_tensor, label

# LSTM model for classification
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, time_steps, input_size]
        out, _ = self.lstm(x)  # out: [batch, time_steps, hidden_size]
        out = out[:, -1, :]    # get last time step output
        out = self.fc(out)     # [batch, num_classes]
        return out

def collate_fn(batch):
    # Pad sequences to max length in batch
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_seqs, labels, lengths

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels, lengths in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    TRAIN_DIR = "../data/mfcc_split_data/train"
    TEST_DIR = "../data/mfcc_split_data/test"
    LABELS_CSV = "../data/labels.csv"

    # Load datasets
    train_dataset = SpikeTrainDataset(TRAIN_DIR, LABELS_CSV)
    test_dataset = SpikeTrainDataset(TEST_DIR, LABELS_CSV)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model params
    input_size = 39
    hidden_size = 128
    num_layers = 2
    num_classes = len(set(pd.read_csv(LABELS_CSV)['label']))  # infer classes from CSV

    # Model
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = eval_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), "lstm_trained.pt")
    print("Training complete and model saved.")

