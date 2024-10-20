import tarfile

# Extract the DAPS dataset
file_path = '../daps.tar.gz'

with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall('daps')

audio_dir = 'extracted_daps/'