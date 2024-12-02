import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import torchvision.transforms as transforms

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


class DAPSAudioDataset_with_cropping(Dataset):
    def __init__(
        self,
        class_mapping,
        spectrogram_dir="precomputed_spectrograms_aug",
        crop_size=64,
        transform=None,
    ):
        self.class_mapping = list(class_mapping.items())
        self.spectrogram_dir = spectrogram_dir
        self.crop_size = crop_size
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize((64, 8000)),
                    transforms.RandomCrop(crop_size),
                ]
            )
        )

    def __getitem__(self, idx):
        spectrogram_path, label = self.class_mapping[idx]
        try:
            spectrogram_tensor = torch.load(spectrogram_path)
            if self.transform:
                spectrogram_tensor = self.transform(spectrogram_tensor)
            return spectrogram_tensor, label
        except Exception as e:
            print(f"Error loading {spectrogram_path}: {e}")
            return None

    def __len__(self):
        return len(self.class_mapping)


import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DAPSAudioDataset_with_multiple_cropping(Dataset):
    def __init__(
        self,
        class_mapping,
        spectrogram_dir="precomputed_spectrograms_aug",
        crop_size=64,
        num_crops=3,
        transform=None,
    ):
        self.class_mapping = list(class_mapping.items())
        self.spectrogram_dir = spectrogram_dir
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize((64, 8000)),
                ]
            )
        )
        self.random_crop = transforms.RandomCrop(crop_size)

        # Extended mapping for multiple crops
        self.extended_mapping = [
            (spectrogram_path, label)
            for spectrogram_path, label in self.class_mapping
            for _ in range(self.num_crops)
        ]

    def __getitem__(self, idx):
        spectrogram_path, label = self.extended_mapping[idx]
        spectrogram_tensor = torch.load(spectrogram_path)
        if self.transform:
            spectrogram_tensor = self.transform(spectrogram_tensor)
        cropped_tensor = self.random_crop(spectrogram_tensor)
        return cropped_tensor, label

    def __len__(self):
        return len(self.extended_mapping)
