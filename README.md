Download the dataset from : https://zenodo.org/records/4660670

Paper on the dataset : https://ccrma.stanford.edu/~gautham/Site/daps_files/mysore-spl2015.pdf

Download conda, create a conda environment, install libraries :

run the following commands for a cpu installation of pytorch :

conda create --name voice_recognition

conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install -c conda-forge librosa








*** Use of each python file ***


1 : Extract_dataset.py
Extracts the DAPS dataset from a tar.gz file to move them into extracted_daps folder.

2 : Generate_spectrograms.py
Generates spectrograms from the extracted audio files for visualization not for training.

3 : Daps_dataset.py
This script creates a dataset that reads audio files, converts them into spectrograms, and labels them based on speaker identity for binary classification tasks.

4 : train_loader.py
Initializes a DataLoader with the custom dataset, preparing it for batching and shuffling the data during the training process.
