import os
import torch
from torch.utils.data import DataLoader
from dap_datasets import DAPSAudioDataset  # Import your dataset class
# from simple_audio_cnn import SimpleAudioCNN  # Import your model class
from spectrograms import generate_spectrograms  # Import the spectrogram generation function

# Define the audio directory
audio_dir = 'daps'  # Adjust this based on your structure
spectrogram_dir = 'spectrograms'  # Directory to save spectrograms
target_folders =['ipad_balcony1']


# Step 1: Generate spectrograms
generate_spectrograms(audio_dir,target_folders, spectrogram_dir=spectrogram_dir)

# Speakers for Class 1
class_1_speakers = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

# Step 2: Create a mapping of audio files to classes
class_mapping = {}

# Load audio files and populate class mapping based on filenames
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            # Get the speaker prefix from the filename
            speaker_prefix = file.split('_')[0]  # Assuming the format is like 'F1_some_other_info.wav'
            # Determine class based on prefix
            if speaker_prefix in class_1_speakers:
                class_mapping[os.path.join(root, file)] = 1  # Class 1
            else:
                class_mapping[os.path.join(root, file)] = 0  # Class 0

print("Class mapping created:")
print(class_mapping)

# Print class distribution
class_0_count = sum(1 for label in class_mapping.values() if label == 0)
class_1_count = sum(1 for label in class_mapping.values() if label == 1)

print(f"Total Class 0 samples: {class_0_count}")
print(f"Total Class 1 samples: {class_1_count}")

# Step 3: Create an instance of the DAPSAudioDataset with the correct class mapping
train_dataset = DAPSAudioDataset(
    class_mapping=class_mapping,
    transform=None  # You can add transformations here if needed
)

# Step 4: Create a DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Example of printing one batch to verify
for batch in train_loader:
    inputs, labels = batch
    print(f"Inputs shape: {inputs.shape}")  # Check shape
    print(f"Labels: {labels}")  # Check labels
    break

# Continue with your model training setup...
