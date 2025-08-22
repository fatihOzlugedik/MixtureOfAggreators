import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np

class MILDataset(Dataset):
    '''Unified MLL MIL dataset for both .pt and .h5 formats'''

    def __init__(self, path_to_data, data_files,label_set, ext="h5"):
        """
        Args:
            path_to_data (str): Path to directory containing patient .pt or .h5 files.
            data_files (pd.DataFrame): DataFrame with columns ['patient_files', 'labels'].
            ext (str): File extension type to use ('pt' or 'h5').
        """
        assert ext in ["pt", "h5"], f"Unsupported file extension: {ext}"
        self.path_to_data = path_to_data
        self.data_files = data_files
        self.ext = ext
        self.label_set = label_set

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
        patient_id = self.data_files.patient_files[0]
        file_name = patient_id + f".{self.ext}"
        file_path = os.path.join(self.path_to_data, file_name)

        if self.ext == "pt":
            features, _, _ = read_ptfile(file_path,self)
        else:
            features, _, _ = read_h5file(file_path,self.label_set)

        return features.shape[1]

    def __getitem__(self, idx):
        patient_id = self.data_files.patient_files[idx]
        

        file_name = patient_id + f".{self.ext}"
        file_path = os.path.join(self.path_to_data, file_name)

        if self.label_set == 'diagnose':
            training_label = self.data_files.diagnose
        elif self.label_set == 'labels':
            training_label = self.data_files.labels

        # Load based on extension
        if self.ext == "pt":
            features,eight_class_label,img_paths= read_ptfile(file_path,label_set=self.label_set)
        else:
            features,eight_class_label,img_paths = read_h5file(file_path,label_set=self.label_set)

        assert features.ndim == 2, f"Expected [N, D], got {features.shape}"
      
        training_label = training_label[idx]

        return features,eight_class_label,training_label,img_paths,patient_id

def read_h5file(file_name, label_set):
    with h5py.File(file_name, 'r') as hf:
        features = hf['features'][()]
        eight_class_label = np.array(hf['labels'])  # Convert to numpy array
        img_paths_raw = hf['img_paths'][()]
        if isinstance(img_paths_raw[0], bytes):
            img_paths = [p.decode('utf-8') for p in img_paths_raw]
        else:
            img_paths = [str(p) for p in img_paths_raw]
    return features, eight_class_label, img_paths

def read_ptfile(file_name, label_set):
    data = torch.load(file_name, map_location=torch.device("cpu"))
    features = data['features']
    eight_class_label = np.array(data['labels'])  # Convert to numpy array
    img_paths = data['image_paths']
    return features, eight_class_label, img_paths