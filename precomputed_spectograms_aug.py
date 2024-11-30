import os
import numpy as np
import librosa
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


def augment_spectrogram(spectrogram):
    noise = torch.randn_like(spectrogram) * 0.01  # Gaussian noise
    augmented = spectrogram + noise

    # Time masking
    time_mask_width = 20
    time_start = torch.randint(0, spectrogram.shape[2] - time_mask_width, (1,)).item()
    augmented[:, :, time_start : time_start + time_mask_width] = 0

    # Frequency masking
    freq_mask_width = 10
    freq_start = torch.randint(0, spectrogram.shape[1] - freq_mask_width, (1,)).item()
    augmented[:, freq_start : freq_start + freq_mask_width, :] = 0

    return augmented


def save_precomputed_spectrograms(
    class_mapping,
    output_dir="precomputed_spectrograms_aug",
    augment_class_1=True,
    augmentations_per_sample=2,
):
    os.makedirs(output_dir, exist_ok=True)
    for audio_path, label in class_mapping.items():
        spectrogram_tensor = preprocess_audio(audio_path)
        if spectrogram_tensor is not None:
            # Save original spectrogram
            file_name = os.path.splitext(os.path.basename(audio_path))[0]
            original_file_path = os.path.join(output_dir, f"{file_name}.pt")
            torch.save(spectrogram_tensor, original_file_path)

            # Augment for class 1 to balance the dataset
            if label == 1 and augment_class_1:
                for aug_idx in range(augmentations_per_sample):
                    augmented_spectrogram = augment_spectrogram(spectrogram_tensor)
                    augmented_file_name = f"{file_name}_aug{aug_idx + 1}.pt"
                    augmented_file_path = os.path.join(output_dir, augmented_file_name)
                    torch.save(augmented_spectrogram, augmented_file_path)

    print("All spectrograms (original and augmented) precomputed and saved.")


save_precomputed_spectrograms(class_mapping)
