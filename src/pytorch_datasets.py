import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np




class DAPSAudioDataset:
    def __init__(self, class_mapping, spectrogram_dir="precomputed_spectrograms"):
        self.class_mapping = list(class_mapping.items())
        self.spectrogram_dir = spectrogram_dir

    def __getitem__(self, idx):
        audio_path, label = self.class_mapping[idx]
        file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".pt"
        spectrogram_path = os.path.join(self.spectrogram_dir, file_name)

        try:
            spectrogram_tensor = torch.load(spectrogram_path)
            return spectrogram_tensor, label
        except Exception as e:
            print(f"Error loading {spectrogram_path}: {e}")
            return None

    def __len__(self):
        return len(self.class_mapping)
