from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.multiprocessing
import torch
import os
import time
import argparse as ap
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from model_train import ModelTrainer
from classifier import cAItomorph         
from dataset import MILDataset     
from plot_confusion import plot_confusion_matrix
from scheduler import BuildScheduler



torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd


def main():
    parser = ap.ArgumentParser()
    # Training parameters
    parser.add_argument(
        '--lr',
        help='learning rate',
        required=False,
        type=float,
        default=5e-5)  
    parser.add_argument(
        '--grad_accum',
        help='gradient accumulation steps',
        required=False,
        type=int,
        default=16)
    parser.add_argument(
        '--scheduler',
        help='scheduler',
        required=False,
        default='ReduceLROnPlateau')  
    parser.add_argument(
        '--ep',
        help='number of epochs to train',
        required=False,
        type=int,
        default=150)             
    parser.add_argument(
        '--es',
        help='early stopping if no decrease in loss for x epochs',
        required=False,
        type=int,
        default=15)          
    parser.add_argument(
        '--wd',
        help='weight decay',
        required=False,
        type=float,
        default=0.01)
    parser.add_argument(
        '--metric',
        help='loss or f1',
        required=False,
        choices=['loss', 'f1'],
        type=str,
        default='loss')  
    parser.add_argument(
        '--seed',
        help='random seed',
        required=False,
        type=int,
        default=38)
    # Model architecture
    parser.add_argument(
        '--arch',
        help='Filters patients with sub-standard sample quality',
        default="Transformer")  
    # Continue from checkpoint                         
    parser.add_argument(
        '--checkpoint',
        help='checkpoint',
        required=False,
        default=None)
    # Data path
    parser.add_argument(
        '--data_path',
        help='data path.',
        default="/lustre/groups/labs/marr/qscd01/workspace/furkan.dasdelen/dino_feature_extractor/DinoBloom-vitb14-features/")
    # Save name
    parser.add_argument(
        '--result_folder',
        help='store folder with custom name',
        required=False)    
        # Mixture-of-Experts / Aggregator setup
    parser.add_argument('--expert_mode', choices=['shared', 'separate', 'shared_adapter'], 
                        default='shared', help="Expert sharing strategy")
    parser.add_argument('--router_style', choices=['topk', 'dense'], 
                        default='topk', help="Routing strategy")
    parser.add_argument('--topk', type=int, default=1, 
                        help="Top-k experts to use when router_style=topk")
    parser.add_argument('--use_local_head', action='store_true', 
                        help="If set, experts have individual heads")
    parser.add_argument('--save_gates', action='store_true',
                        help="Save router gate activations during training/validation")
                           
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # store results in target folder
    checkpoint = args.checkpoint
    start = time.time()

    data_path = args.data_path
    backbone_name = data_path.split('/')[-1]

    RESULT_FOLDER_ROOT = f"Results_5fold_testfixed_{backbone_name}_{args.arch}_{args.expert_mode}_{args.router_style}_topk{args.topk}_localhead{args.use_local_head}_seed{seed}"
    def get_unique_folder(base_folder):
        counter = 1
        new_folder = base_folder
        
        while os.path.exists(new_folder):
            new_folder = f"{base_folder}{counter}"
            counter += 1
        
        return new_folder

    if not os.path.exists(RESULT_FOLDER_ROOT):
        os.makedirs(RESULT_FOLDER_ROOT)
    else:
        if checkpoint is None:
            unique_folder = get_unique_folder(RESULT_FOLDER_ROOT)
            os.makedirs(unique_folder)
            RESULT_FOLDER_ROOT = unique_folder
    
    RESULT_FOLDER_ROOT = Path(RESULT_FOLDER_ROOT)
    print('Results are saved in: ',RESULT_FOLDER_ROOT)

    # 2: Dataset
    print("")
    print('Initialize datasets...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)
    
    datasets = {}
    
    csv_root = "data_cross_val_8_classes"

    label_to_diagnose = pd.read_csv(os.path.join(csv_root,"label_to_diagnose.csv"))
    #m_data = pd.read_csv(os.path.join(csv_root,"age_sex_data.csv"))

    class_count = len(label_to_diagnose)

    print('Reading files from: ',os.path.join(csv_root,f'data_fold_1'))
    t_files = pd.read_csv(os.path.join(csv_root,f'data_fold_1',"train.csv"))
    v_files = pd.read_csv(os.path.join(csv_root,f'data_fold_1',"val.csv"))
    train_val_files = pd.concat([t_files, v_files], ignore_index=True)
    test_files = pd.read_csv(os.path.join(csv_root,f'data_fold_1',"test.csv"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_index, val_index) in enumerate(skf.split(train_val_files, train_val_files['diagnose'])):
        train_files = train_val_files.iloc[train_index].reset_index(drop=True)
        val_files = train_val_files.iloc[val_index].reset_index(drop=True)

        RESULT_FOLDER = RESULT_FOLDER_ROOT / f'fold_{fold}'
        RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

        datasets['train'] = MILDataset(data_path, train_files)
        datasets['val'] = MILDataset(data_path, val_files)
        datasets['test'] = MILDataset(data_path, test_files)
        
        embedding_dim = datasets['train'].get_dimension()
        print(f"Embedding dimension: {embedding_dim}")
        
        # Initialize dataloaders
        print("Initialize dataloaders...")
        dataloaders = {}
        num_workers = 4

        dataloaders['train'] = DataLoader(datasets['train'], batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
        dataloaders['val'] = DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        dataloaders['test'] = DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        print("Dataloaders are ready..")
        
        model = cAItomorph(
            class_count=class_count,
            arch=args.arch,
            embedding_dim=embedding_dim,
            expert_mode=args.expert_mode,
            router_style=args.router_style,
            topk=args.topk,
            use_local_head=args.use_local_head,
            save_gates=args.save_gates
        )

        if checkpoint is not None:
            pre = torch.load(checkpoint)
            vit_state_dict = {k.replace("module.", ""): v for k, v in pre.items()}
            model.load_state_dict(vit_state_dict, strict=True)
            print(f"Using weights from {checkpoint}")
        
        if(ngpu > 1):
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        print(model.eval())
        print("Setup complete.")
        print("")


        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.wd)
        sched_builder = BuildScheduler(optimizer, args)
        print("Using scheduler:", sched_builder.name)    
        
        # launch training
        train_obj = ModelTrainer(
            model=model,
            dataloaders=dataloaders,
            epochs=int(args.ep),
            optimizer=optimizer,
            sched_builder=sched_builder,
            class_count=class_count,
            device=device,
            early_stop=int(args.es),
            save_path=RESULT_FOLDER)
        print("Starting training")

        model, conf_matrix = train_obj.launch_training()


        np.save(RESULT_FOLDER / 'test_conf_matrix.npy', conf_matrix)
        plot_confusion_matrix(conf_matrix, RESULT_FOLDER, label_to_diagnose)

        #torch.save(model, RESULT_FOLDER / 'best_model.pt')

        end = time.time()
        runtime = end - start
        time_str = str(int(runtime // 3600)) + "h" + str(int((runtime %
                                                            3600) // 60)) + "min" + str(int(runtime % 60)) + "s"
        
        # other parameters
        print("")
        print("------------------------Final report--------------------------")
        print('Runtime', time_str)
        print('Fold', fold)
        print('Architecture', args.arch)
        print('Data path', args.data_path)
        print('Result folder', RESULT_FOLDER)
        print('Scheduler', args.scheduler)
        print('Weight decay', args.wd)
        print('Early stopping', args.es)
        print('Gradient accumulation', args.grad_accum)  
        print('Seed', seed)
        print('max. Epochs', args.ep)
        print('Learning rate', args.lr)
    

if __name__ == "__main__":
    main()
    