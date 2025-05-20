import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from stork.models import RecurrentSpikingModel
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from stork.connections import DenseConnection
from stork import loss_stacks
from sklearn.metrics import accuracy_score

def predict_single_file(model, filepath, label_csv, nb_time_steps, nb_features):
    # Load and preprocess the file
    device = torch.device('cpu')
    spike_train = np.load(filepath)
    
    # Pad/truncate to expected dimensions
    if spike_train.shape[0] > nb_time_steps:
        spike_train = spike_train[:nb_time_steps]
    else:
        spike_train = np.pad(spike_train, ((0, nb_time_steps - spike_train.shape[0]), (0, 0)))
    
    if spike_train.shape[1] > nb_features:
        spike_train = spike_train[:, :nb_features]
    else:
        spike_train = np.pad(spike_train, ((0, 0), (0, nb_features - spike_train.shape[1])))
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(spike_train, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model.forward_pass(input_tensor, cur_batch_size=1)
        probs = torch.exp(model.loss_stack.log_py_given_x(output))
        pred = model.loss_stack.predict(output).item()
    
    # Get true label if available
    filename = os.path.basename(filepath)
    label_df = pd.read_csv(label_csv)
    true_label = label_df[label_df['filename'] == filename]['label'].values[0] if filename in label_df['filename'].values else None
    
    return {
        'filename': filename,
        'prediction': pred,
        'confidence': probs[0,pred].item(),
        'true_label': true_label,
        'all_probabilities': probs.cpu().numpy()[0]
    }


# Dataset class reused
class SpikeTrainDataset(Dataset):
    def __init__(self, data_dir, label_csv, nb_time_steps, nb_features):
        self.filepaths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.labels_df = pd.read_csv(label_csv)
        self.nb_time_steps = nb_time_steps
        self.nb_features = nb_features
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        spike_train = np.load(self.filepaths[idx])
        
        if spike_train.shape[0] > self.nb_time_steps:
            spike_train = spike_train[:self.nb_time_steps]
        else:
            spike_train = np.pad(spike_train, ((0, self.nb_time_steps - spike_train.shape[0]), (0, 0)))
        
        if spike_train.shape[1] > self.nb_features:
            spike_train = spike_train[:, :self.nb_features]
        else:
            spike_train = np.pad(spike_train, ((0, 0), (0, self.nb_features - spike_train.shape[1])))
        
        label = self.labels_df[self.labels_df['filename'] == os.path.basename(self.filepaths[idx])]['label'].values[0]
        return torch.tensor(spike_train, dtype=torch.float32), int(label)

def load_model(checkpoint_path, batch_size, device):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Rebuild model
    model = RecurrentSpikingModel(
        batch_size=batch_size,
        nb_time_steps=config['nb_time_steps'],
        nb_inputs=config['nb_features'],
        device=device
    )

    # Build architecture
    input_group = InputGroup(config['nb_features'], name="Input")
    hidden_group = LIFGroup(config['nb_hidden'], name="Hidden")
    output_group = ReadoutGroup(config['nb_outputs'])

    model.add_group(input_group)
    model.add_group(hidden_group)
    model.add_group(output_group)

    model.add_connection(DenseConnection(input_group, hidden_group))
    model.add_connection(DenseConnection(hidden_group, output_group))

    model.configure(
        input=input_group,
        output=output_group,
        loss_stack=loss_stacks.MaxOverTimeCrossEntropy(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        time_step=1e-3
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config

def evaluate_model(model, test_loader, device):
    model.eval()
    test_losses = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model.forward_pass(batch_X, len(batch_X))  # shape: (batch_size, num_classes)

            # Compute loss
            loss = model.get_total_loss(output, batch_y)
            test_losses.append(loss.item())

            # Get predictions
            pred_labels = model.loss_stack.predict(output)  # [batch_size]  # <--- Use this if available
            all_preds.extend(pred_labels.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    # Compute average loss and accuracy
    avg_loss = np.mean(test_losses)
    acc = accuracy_score(all_targets, all_preds)

    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    # Paths
    checkpoint_path = "spikehear_model_state.pth"
    test_data_dir = "../data/mfcc_split_data/test"
    label_csv = "../data/labels.csv"

    # Params
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, config = load_model(checkpoint_path, batch_size, device)

    # Prepare test set
    test_dataset = SpikeTrainDataset(test_data_dir, label_csv, config['nb_time_steps'], config['nb_features'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Evaluate
    evaluate_model(model, test_loader, device)
    
    # Test on a single file
    result = predict_single_file(
        model,
        '../data/mfcc_split_data/test/2374_spike.npy',
        '../data/labels.csv',
        nb_time_steps=100,
        nb_features=10
    )

    print(f"""
    File: {result['filename']}
    Predicted: {result['prediction']}
    Confidence: {result['confidence']:.2%}
    True Label: {result['true_label']}
    All Probabilities: {result['all_probabilities']}
    """)

