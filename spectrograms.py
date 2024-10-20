import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Function to generate and save spectrograms for specific folders
def generate_spectrograms(audio_dir, target_folders, spectrogram_dir='spectrograms'):
    os.makedirs(spectrogram_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for folder in target_folders:
        folder_path = os.path.join(audio_dir, folder)  # Build the folder path
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist, skipping.")
            continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".wav"):
                    audio_path = os.path.join(root, file)
                    try:
                        # Load the audio file
                        audio, sr = librosa.load(audio_path, sr=None)

                        # Convert audio to a spectrogram using STFT
                        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

                        # Prepare the spectrogram file path
                        base_name = file.replace('.wav', '.png')  # Save as PNG image
                        spectrogram_file = os.path.join(spectrogram_dir, base_name)

                        # Plot and save the spectrogram
                        plt.figure(figsize=(10, 4))
                        plt.imshow(spectrogram, aspect='auto', origin='lower')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title('Spectrogram')
                        plt.xlabel('Time (frames)')
                        plt.ylabel('Frequency bins')
                        plt.tight_layout()
                        plt.savefig(spectrogram_file)
                        plt.close()

                        print(f"Saved spectrogram for {file} in {folder}")

                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")

# Example call (can be removed or moved into another file):
# generate_spectrograms('path_to_audio_folder', ['class_0', 'class_1'])
