import librosa
import numpy as np
import os
from tqdm import tqdm

input_dir = "recordings"
output_dir = "mfcc_spike_trains"
os.makedirs(output_dir, exist_ok=True)

threshold = -10  # Adjust depending on the value range of your MFCCs

for idx, filename in enumerate(tqdm(sorted(os.listdir(input_dir)))):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)

        # Load audio
        y, sr = librosa.load(filepath, sr=None)
        y = y / max(abs(y))  # Normalize

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # You can use 13–40 coefficients

        # Optional: Append delta and delta-delta
        delta = librosa.feature.delta(mfcc, width=3)
        delta2 = librosa.feature.delta(mfcc, order=2, width=3)
        combined = np.vstack([mfcc, delta, delta2])  # Shape: (39, T)

        # Convert to spike train (binary)
        spike_train = (combined > threshold).astype(np.uint8)

        # Save spike train
        save_path = os.path.join(output_dir, f"{idx}_spike.npy")
        np.save(save_path, spike_train)

import matplotlib.pyplot as plt
plt.imshow(spike_train, aspect='auto', cmap='inferno')
plt.title(filename)
plt.show()
