import tarfile

# Extract the DAPS dataset
file_path = '../daps.tar.gz'

with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall('extracted_daps')

# Directory containing the extracted dataset
audio_dir = 'extracted_daps/'