import librosa
import numpy as np
import os
from tqdm import tqdm

input_dir = "recordings"
output_dir = "spike_trains"
os.makedirs(output_dir, exist_ok=True)

#threshold = -20  

for idx, filename in enumerate(tqdm(sorted(os.listdir(input_dir)))):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)

        # Load audio
        y, sr = librosa.load(filepath, sr=None)

        y = y / max(abs(y))
        # Convert to Mel spectrogram
        # Convert to Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Use percentile threshold instead of fixed dB
        threshold = np.percentile(mel_db, 70)  # You can try 70–85 and tune
        spike_train = (mel_db > threshold).astype(np.uint8)

        # Threshold-based spike train
        #spike_train = (mel_db > threshold).astype(np.uint8)  # Save as 0/1 uint8

        # Save as .npy
        save_path = os.path.join(output_dir, f"{idx}_spike.npy")
        np.save(save_path, spike_train)

import matplotlib.pyplot as plt
plt.imshow(spike_train, aspect='auto', origin='lower')
plt.title(filename)
plt.show()


