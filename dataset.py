import torch
from torch.utils.data import Dataset
import h5py
import os

class MILDataset(Dataset):
    '''Unified MLL MIL dataset for both .pt and .h5 formats'''

    def __init__(self, path_to_data, data_files, ext="h5"):
        """
        Args:
            path_to_data (str): Path to directory containing patient .pt or .h5 files.
            data_files (pd.DataFrame): DataFrame with columns ['patient_names', 'labels'].
            ext (str): File extension type to use ('pt' or 'h5').
        """
        assert ext in ["pt", "h5"], f"Unsupported file extension: {ext}"
        self.path_to_data = path_to_data
        self.data_files = data_files
        self.ext = ext

    def __len__(self):
        return len(self.data_files)
    
    def get_dimension(self):
        """
        Returns:
            int: The number of features in the dataset.
        """
        if len(self.data_files) == 0:
            return 0
        
        # Load the first file to determine feature dimension
        patient_id = self.data_files.patient_name[0]
        file_name = patient_id + f".{self.ext}"
        file_path = os.path.join(self.path_to_data, file_name)

        if self.ext == "pt":
            features, _, _ = read_ptfile(file_path)
        else:
            features, _, _ = read_h5file(file_path)

        return features.shape[1]

    def __getitem__(self, idx):
        patient_id = self.data_files.patient_name[idx]
        file_name = patient_id + f".{self.ext}"
        file_path = os.path.join(self.path_to_data, file_name)

        # Load based on extension
        if self.ext == "pt":
            features, label, img_paths = read_ptfile(file_path)
        else:
            features, label, img_paths = read_h5file(file_path)

        assert features.ndim == 2, f"Expected [N, D], got {features.shape}"

        # Optionally override label from CSV if needed (only if trustworthy)
        label = self.data_files.labels[idx]

        return features, label, img_paths, patient_id

def read_h5file(file_name):
    with h5py.File(file_name, 'r') as hf:
        features = hf['features'][()]
        label = hf['labels'][()]
        img_paths_raw = hf['img_paths'][()]
        if isinstance(img_paths_raw[0], bytes):
            img_paths = [p.decode('utf-8') for p in img_paths_raw]
        else:
            img_paths = [str(p) for p in img_paths_raw]
    return features, label, img_paths

def read_ptfile(file_name):
    data = torch.load(file_name, map_location=torch.device("cpu"))
    features = data['features']
    label = ['dummy']
    img_paths = ['dummy']
    return features, label, img_paths