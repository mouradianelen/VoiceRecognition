import librosa
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class DAPSAudioDataset(Dataset):
    def __init__(self, class_mapping, transform=None):
        """
        Args:
            class_mapping (dict): A dictionary mapping audio file paths to their class labels (0 or 1).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.class_mapping = class_mapping  # Mapping of audio files to class labels
        self.audio_files = list(class_mapping.keys())
        self.labels = list(class_mapping.values())
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None)

            # Convert to spectrogram
            spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

            # Normalize or augment the spectrogram if transformations are provided
            if self.transform:
                spectrogram = self.transform(spectrogram)

            # Convert spectrogram to tensor and reshape for CNN input
            spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

            # Pad or truncate the spectrogram to a fixed size if needed
            desired_shape = (1, 1024, 128)  # Example target shape
            spectrogram_tensor = self._pad_or_truncate(spectrogram_tensor, desired_shape)

            return spectrogram_tensor, label

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None  # You can return a default value or raise an error

    def _pad_or_truncate(self, tensor, desired_shape):
        """
        Pad or truncate the tensor to the desired shape.
        
        Args:
            tensor (torch.Tensor): Input tensor to pad or truncate.
            desired_shape (tuple): Desired shape for the output tensor.

        Returns:
            torch.Tensor: Padded or truncated tensor.
        """
        # If the tensor's shape does not match the desired shape
        if tensor.shape[1] < desired_shape[1]:  # If shorter, pad
            padding = desired_shape[1] - tensor.shape[1]
            tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0)
        elif tensor.shape[1] > desired_shape[1]:  # If longer, truncate
            tensor = tensor[:, :, :desired_shape[1]]
        
        return tensor
