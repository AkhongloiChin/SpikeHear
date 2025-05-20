import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from stork.models import RecurrentSpikingModel
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from stork.connections import DenseConnection
#from stork.connections import Connection as DenseConnection
from stork import loss_stacks

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
        elif spike_train.shape[0] < self.nb_time_steps:
            spike_train = np.pad(spike_train, ((0, self.nb_time_steps - spike_train.shape[0]), (0, 0)))
        
        if spike_train.shape[1] > self.nb_features:
            spike_train = spike_train[:, :self.nb_features]
        elif spike_train.shape[1] < self.nb_features:
            spike_train = np.pad(spike_train, ((0, 0), (0, self.nb_features - spike_train.shape[1])))
        
        label = self.labels_df[self.labels_df['filename'] == os.path.basename(self.filepaths[idx])]['label'].values[0]
        return torch.tensor(spike_train, dtype=torch.float32), int(label)

def main():
    # Parameters
    batch_size = 32
    nb_time_steps = 100 
    nb_hidden = 128    
    nb_outputs = 10      
    time_step = 1e-3    
    nb_features = 10  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = RecurrentSpikingModel(
        batch_size=batch_size,
        nb_time_steps=nb_time_steps,
        nb_inputs=nb_features,
        device=device
    )

    # Input group (spike sources)
    input_group = InputGroup(nb_features, name="Input")
    model.add_group(input_group)

    # Hidden layer
    hidden_group = LIFGroup(nb_hidden, name="Hidden")
    model.add_group(hidden_group)

    # Output layer (readout)
    output_group = ReadoutGroup(nb_outputs)#, name="Output")
    model.add_group(output_group)

    # Create connections
    input_to_hidden = DenseConnection(input_group, hidden_group)
    model.add_connection(input_to_hidden)

    hidden_to_output = DenseConnection(hidden_group, output_group)
    model.add_connection(hidden_to_output)

    # Configure the model
    model.configure(
        input=input_group,
        output=output_group,
        loss_stack = loss_stacks.MaxOverTimeCrossEntropy(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        time_step=time_step
    )

    train_dataset = SpikeTrainDataset("../data/mfcc_split_data/train", "../data/labels.csv", 
                                    nb_time_steps, nb_features)
    test_dataset = SpikeTrainDataset("../data/mfcc_split_data/test", "../data/labels.csv",
                                   nb_time_steps, nb_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    nb_epochs = 300
    # Train the model
    for epoch in range(nb_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            output = model.forward_pass(batch_X, len(batch_X))
            
            # Compute loss and backprop
            total_loss = model.get_total_loss(output, batch_y)
            model.optimizer_instance.zero_grad()
            total_loss.backward()
            model.optimizer_instance.step()
            model.apply_constraints()

    torch.save(model, 'spikehear_model.pth')
    
    torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'nb_features': nb_features,
        'nb_hidden': nb_hidden,
        'nb_outputs': nb_outputs,
        'nb_time_steps': nb_time_steps
    }
    }, 'spikehear_model_state.pth')
    
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
