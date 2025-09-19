from torch.utils.data import DataLoader
import torch.multiprocessing
import torch
import os
import time
import argparse as ap
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import networkx as nx
import gc

from infer import ModelInfer
from classifier import ClassifierWrapper     
from dataset import MILDataset 
from plot_confusion import plot_confusion_matrix

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # 1: Parse arguments
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to the config file'
    )

    args = parser.parse_args()
    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # store results in target folder

    start = time.time()

    RESULT_FOLDER = f"Results_fold_{config['dataset_name']}_{config['arch']}"
    def get_unique_folder(base_folder):
        counter = 1
        new_folder = base_folder
        
        while os.path.exists(new_folder):
            new_folder = f"{base_folder}-{counter}"
            counter += 1
        
        return new_folder

    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    else:
        unique_folder = get_unique_folder(RESULT_FOLDER)
        os.makedirs(unique_folder)
        RESULT_FOLDER = unique_folder
    
    RESULT_FOLDER = Path(RESULT_FOLDER)
    print("Results will be saved in: ", RESULT_FOLDER)
    # 2: Dataset
    # Initialize datasets, dataloaders, ...
    print("")
    print('Initialize datasets...')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)
    
    ncpu = os.cpu_count()
    print("ncpu="+str(ncpu))
    
    datasets = {}

    data_path = config['data_path']
    ext = config['ext']
    
    label_to_diagnose = pd.read_csv(config['label_to_diagnose'])

    #class_count = len(label_to_diagnose)
    class_count = 8
    
    test_files = pd.read_csv(config['test_files'])

    if len(test_files) != len(list(Path(data_path).glob(f"*.{ext}"))):
        print("Warning: The number of test files does not match the expected number based on the data path.")
        print(f"Expected: {len(test_files)} Found: {len(list(Path(data_path).glob(f'*.{ext}')))}")
        print("First extract features using extract_features.py to ensure the dataset is ready.")
        return 0

    datasets['test'] = MILDataset(data_path, test_files)

    # Initialize dataloaders
    print("Initialize dataloaders...")
    dataloaders = {}
    
    num_workers = 4
     
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=1, shuffle = False, num_workers=num_workers, pin_memory=True)

    print("Dataloaders are ready..")

    model_path = config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = ClassifierWrapper(
        class_count=class_count,
        arch=config['arch'],
        embedding_dim=datasets['test'].get_dimension()
    )

    print(model.eval())
    #print(model.state_dict().keys())
    
    pre = torch.load(model_path)
    #vit_state_dict = {k.replace("module.", ""): v for k, v in pre.items()}
    model.load_state_dict(pre, strict=True)
    
    model = model.to(device)

    # launch training
    infer_obj = ModelInfer(
        model=model,
        dataloaders=dataloaders,
        class_count=class_count,
        device=device,
        save_path = RESULT_FOLDER)
    print("Starting inferring")
    model, conf_matrix= infer_obj.launch_infering()
    
    
    # 4: aftermath
    # save confusion matrix from test set, all the data , model, print parameters
    
    np.save(os.path.join(RESULT_FOLDER, 'test_conf_matrix.npy'), conf_matrix)
    plot_confusion_matrix(conf_matrix, RESULT_FOLDER, label_to_diagnose)
    
    end = time.time()
    runtime = end - start
    time_str = str(int(runtime // 3600)) + "h" + str(int((runtime %
                                                          3600) // 60)) + "min" + str(int(runtime % 60)) + "s"
    
    # other parameters
    print("")
    print("------------------------Final report--------------------------")
    print('Runtime', time_str)
    

if __name__ == "__main__":
    main()
    
