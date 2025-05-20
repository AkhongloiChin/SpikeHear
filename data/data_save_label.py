import pandas as pd
import os
label_data = []

input_dir = 'recordings'
for idx, filename in enumerate(sorted(os.listdir(input_dir))):
    if filename.endswith(".wav"):
        label = int(filename.split("_")[0])  
        label_data.append((f"{idx}_spike.npy", label))

df = pd.DataFrame(label_data, columns=["filename", "label"])
df.to_csv("labels.csv", index=False)
