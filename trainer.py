import os
from torch.utils.data import DataLoader
from dap_datasets import DAPSAudioDataset  
from spectrograms import generate_spectrograms  

audio_dir = 'daps'  
spectrogram_dir = 'spectrograms'  
target_folders =['ipad_balcony1']


generate_spectrograms(audio_dir,target_folders, spectrogram_dir=spectrogram_dir)

class_1_speakers = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

class_mapping = {}

for root, dirs, files in os.walk(audio_dir):
    for file in files:
        # Skip files that start with "._" or are not ".wav" files
        if file.startswith("._") or not file.endswith(".wav"):
            continue

        speaker_prefix = file.split("_")[0]
        if speaker_prefix in class_1_speakers:
            class_mapping[os.path.join(root, file)] = 1
        else:
            class_mapping[os.path.join(root, file)] = 0

print("Class mapping created:")
print(class_mapping)

# Print class distribution
class_0_count = sum(1 for label in class_mapping.values() if label == 0)
class_1_count = sum(1 for label in class_mapping.values() if label == 1)

print(f"Total Class 0 samples: {class_0_count}")
print(f"Total Class 1 samples: {class_1_count}")

train_dataset = DAPSAudioDataset(
    class_mapping=class_mapping,
    transform=None  
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    inputs, labels = batch
    print(f"Inputs shape: {inputs.shape}")  
    print(f"Labels: {labels}")  
    break
