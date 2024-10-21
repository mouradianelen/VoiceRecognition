1 : Extract_dataset.py
Purpose: Extracts the DAPS dataset from a tar.gz file.
Description: This script unpacks the compressed archive and extracts all audio files into a specified directory.

2 : Generate_spectrograms.py
Purpose: Generates spectrograms from the extracted audio files.
Description: The script processes audio files and converts them into spectrograms, which are saved in a specific folder for later use in training.

3 : Daps_dataset.py
Purpose: Defines a custom PyTorch dataset class for handling audio files.
Description: This script creates a dataset that reads audio files, converts them into spectrograms, and labels them based on speaker identity for binary classification tasks.

4 : train_loader.py
Purpose: Sets up the DataLoader for training.
Description: Initializes a DataLoader with the custom dataset, preparing it for batching and shuffling the data during the training process.