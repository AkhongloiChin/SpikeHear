import os
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

input_dir = "mfcc_spike_trains"
output_dir = "mfcc_split_data"
label_csv = "labels.csv"

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

df = pd.read_csv(label_csv)

spike_files = set(os.listdir(input_dir))
df = df[df["filename"].isin(spike_files)]

filenames = df["filename"].tolist()
labels = df["label"].tolist()

label_counts = df["label"].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df = df[df["label"].isin(valid_labels)]

filenames = df["filename"].tolist()
labels = df["label"].tolist()

train_files, test_files = train_test_split(
    filenames,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

def copy_files(file_list, dest_dir):
    for file in file_list:
        src = os.path.join(input_dir, file)
        dst = os.path.join(dest_dir, file)
        shutil.copy(src, dst)

copy_files(train_files, train_dir)
copy_files(test_files, test_dir)

print(f"Split {len(filenames)} files into {len(train_files)} train and {len(test_files)} test.")
