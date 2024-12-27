import os
import numpy as np
import librosa
from torch.utils.data import DataLoader, random_split
from src.pytorch_datasets import DAPSAudioDataset
import torch

audio_dir = "daps"

class_1_speakers = ["f1", "f7", "f8", "m3", "m6", "m8"]

class_mapping = {}

for root, dirs, files in os.walk(audio_dir):
    for file in files:
        # Skip files that start with "._" or are not ".wav" files
        if file.startswith("._") or not file.endswith(".wav"):
            continue

        audio_path = os.path.join(root, file)
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            speaker_prefix = file.split("_")[0]
            if speaker_prefix in class_1_speakers:
                class_mapping[audio_path] = 1
            else:
                class_mapping[audio_path] = 0
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

print("Class mapping created:")
# print(class_mapping)

class_0_count = sum(1 for label in class_mapping.values() if label == 0)
class_1_count = sum(1 for label in class_mapping.values() if label == 1)

print(f"Total Class 0 samples: {class_0_count}")
print(f"Total Class 1 samples: {class_1_count}")


def preprocess_audio(audio_path, max_length=16000):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        if audio is None or len(audio) == 0:
            return None

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=8000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        normalized_spectrogram = (
            mel_spectrogram_db - np.mean(mel_spectrogram_db)
        ) / np.std(mel_spectrogram_db)

        target_length = max_length
        if normalized_spectrogram.shape[1] > target_length:
            normalized_spectrogram = normalized_spectrogram[:, :target_length]
        else:
            padding = target_length - normalized_spectrogram.shape[1]
            normalized_spectrogram = np.pad(
                normalized_spectrogram, ((0, 0), (0, padding)), mode="constant"
            )

        spectrogram_tensor = torch.tensor(
            normalized_spectrogram, dtype=torch.float32
        ).unsqueeze(0)
        return spectrogram_tensor
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def save_precomputed_spectrograms(class_mapping, output_dir="precomputed_spectrograms"):
    os.makedirs(output_dir, exist_ok=True)
    for audio_path, label in class_mapping.items():
        spectrogram_tensor = preprocess_audio(audio_path)
        if spectrogram_tensor is not None:

            file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".pt"
            torch.save(spectrogram_tensor, os.path.join(output_dir, file_name))
    print("All spectrograms precomputed and saved.")


save_precomputed_spectrograms(class_mapping)
