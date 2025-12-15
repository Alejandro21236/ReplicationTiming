

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import math
import random
import time
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
from dataset.mydataset import WSIDataset
from models.mymodel import WSIClassifier
import utils
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import re
import gc

parser = argparse.ArgumentParser(description='RT Prediction from Histopathology WSIs')
parser.add_argument('--arch', default='WSIClassifier', type=str,
                    help='architecture')
parser.add_argument('--mil', default='att', type=str,
                    help='type of mil')
parser.add_argument('--data', default='WSIDataset', type=str,
                    help='dataset')    
parser.add_argument('--seed', default=7, type=int,
                    help='torch random seed')                
parser.add_argument('--nfold', default=5, type=int,
                    help='number of fold')                         
parser.add_argument('--fold_idx', default=0, type=int,
                    help='fold idx')                    
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='batch size')          
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs')   
parser.add_argument('--patch_dim', default=1536, type=int,
                    help='patch dim')   
parser.add_argument('--cell_dim', default=1280, type=int,
                    help='cell dim')         
parser.add_argument('--hidden_dim1', default=512, type=int,
                    help='hidden layer dim')        
parser.add_argument('--hidden_dim2', default=512, type=int,
                    help='hidden layer dim')            
parser.add_argument('--hidden_dim3', default=256, type=int,
                    help='hidden layer dim')              
parser.add_argument('--num_classes', default=1, type=int,
                    help='number of classes')                 
parser.add_argument('--nsamples', default=15000, type=int,
                    help='number of samples to pick randomly')     
parser.add_argument('--distance_metric', default='euclidean', type=str, 
                    help='distance metric for spatial bias')  
parser.add_argument('--use_spatial_bias', action='store_true', 
                    default=False, help='use spatial bias')  
parser.add_argument('--use_cell_ratios', action='store_true', 
                    default=False, help='use cell ratios')  
parser.add_argument('--use_patch_embeddings', action='store_true', default=True,
                   help='use patch embeddings in the model') 
parser.add_argument('--patch_embeddings_only', action='store_true', default=False,
                   help='use patch embeddings only in the model')  
parser.add_argument('--use_linearprob', action='store_true', default=False,
                   help='use patches filtered by linear probe')   
parser.add_argument('--tumor_only', action='store_true', default=False,
                   help='use patches from tumor region only')   
parser.add_argument('--use_single_cell', default=-1, type=int, 
                    help='use only one type of cell')    
parser.add_argument('--use_extended_dataset', action='store_true', default=False,
                   help='use extended dataset')                        
parser.add_argument('--code', default='testing', type=str,
                    help='exp code')                        
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='accuracy threshold') 
parser.add_argument('--lr', default=5e-5, type=float, 
                    help='init learning rate')  
parser.add_argument('--dropout', default=0.1, type=float, 
                    help='droupout rate')     
parser.add_argument('--regtype', default='l2', type=str, 
                    help='regularization type l1/l2') 
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--weighted_sample', action='store_true', 
                    default=False, help='enable weighted sampling')                    
parser.add_argument('--freq', default=100, type=int, 
                    help='training log display frequency')                              
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')  
parser.add_argument('--pretrained', default='', type=str, 
                    help='pretrained model for validate')                        
parser.add_argument('--pw', default=1, type=int, 
                    help='factor to multiply ratio of positive class by it')        
parser.add_argument('--patience', default=20, type=int, 
                    help='early stopping patience') 
parser.add_argument('--stop_epoch', default=5, type=int, 
                    help='start epoch to activate early stopping counting') 
parser.add_argument('--monitor', default='loss', type=str, 
                    help='value to monitor for early stopping')                     
parser.add_argument('--num_workers', default=0, type=int, 
                    help='number of workers')                     
parser.add_argument('--projection_dim', default=1, type=int, 
                    help='attention weights projection dimension')                     
parser.add_argument('--aggregation_method', default='norm', type=str, 
                    help='aggregation method')                                          
parser.add_argument('--encoder', default='uni_v2', type=str, 
                    help='patch encoder')                     
parser.add_argument('--t', default=100, type=int, 
                    help='t for keyset')                                          
parser.add_argument('--organ', default='brca', type=str, 
                    help='name of organ') 
parser.add_argument('--mutation', default='rt mean', type=str, 
                    help='name of mutation')
parser.add_argument('--log_dir', default="/fs/scratch/PAS2942/Alejandro/RT/logs", type=str, help='directory to store log files')

# New argument for resume functionality
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from latest checkpoint if available')

# Map slide_filename to matching npy file
def find_npy(slide_filename, npy_files, embd_dir):
    # Skip missing or corrupted entries (floats, NaN, None)
    if not isinstance(slide_filename, str):
        return None

    slide_filename = slide_filename.strip()
    if slide_filename == "":
        return None

    base = slide_filename.replace(".svs", "")

    for f in npy_files:
        if f.startswith(base):
            return os.path.join(embd_dir, f)

    return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decrement(match):
    num = int(match.group(1)) - 1
    return f"{num}ep_checkpoint.pth.tar"

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory"""
    if not os.path.exists(checkpoint_dir):
        return None, -1
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*ep_checkpoint.pth.tar"))
    if not checkpoint_files:
        return None, -1
    
    latest_epoch = -1
    latest_checkpoint = None
    
    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        match = re.search(r'(\d+)ep_checkpoint\.pth\.tar', filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = checkpoint_file
    
    return latest_checkpoint, latest_epoch


def find_existing_log_file(log_dir, code, fold_idx):
    """Find existing log file for the same experiment"""
    if not os.path.exists(log_dir):
        return None
    
    # Look for log files matching the pattern
    pattern = f"training_*_{code}_{fold_idx}.log"
    log_files = glob.glob(os.path.join(log_dir, pattern))
    
    if log_files:
        # Return the most recent log file
        return max(log_files, key=os.path.getmtime)
    
    return None

def copy_previous_logs(existing_log_file, new_log_file):
    """Copy content from existing log file to new log file"""
    if existing_log_file and os.path.exists(existing_log_file):
        with open(existing_log_file, 'r') as old_file:
            content = old_file.read()
        
        # Remove the handlers to avoid duplicate logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Reconfigure logging with the new file, appending previous content
        with open(new_log_file, 'w') as new_file:
            new_file.write(content)
            if content and not content.endswith('\n'):
                new_file.write('\n')
            new_file.write(f"\n{'='*50}\n")
            new_file.write(f"RESUMING TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            new_file.write(f"{'='*50}\n\n")
        
        # Reconfigure logging to append to the file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(new_log_file, mode='a'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        logging.info("Previous training logs loaded successfully")
        return True
    return False

def setup_logging(args, resume_info=None):
    """Set up logging configuration"""
    # Create log directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(args.log_dir, f'training_{timestamp}_{args.code}_{args.fold_idx}.log')
    
    # If resuming, try to find and copy previous logs
    if args.resume and resume_info is None:
        existing_log = find_existing_log_file(args.log_dir, args.code, args.fold_idx)
        if copy_previous_logs(existing_log, log_filename):
            logging.info("Resumed from existing logs")
        else:
            # Configure logging normally if no existing logs found
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[
                    logging.FileHandler(log_filename),
                    logging.StreamHandler()
                ]
            )
            logging.info(f"Starting new training run at {timestamp} (resume requested but no previous logs found)")
    else:
        # Configure logging normally
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        if resume_info:
            logging.info(f"Resuming training from epoch {resume_info['epoch']} at {timestamp}")
        else:
            logging.info(f"Starting new training run at {timestamp}")
    
    # Log initial setup information
    logging.info("Command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    return log_filename

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model and optimizer state from checkpoint"""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    resume_info = {
        'epoch': checkpoint['epoch'] + 1,  # Next epoch to train
        'best_metric': checkpoint.get('best_metric', 0),
        'early_stopping_counter': checkpoint.get('early_stopping_counter', 0)
    }
    
    logging.info(f"Checkpoint loaded. Resuming from epoch {resume_info['epoch']}")
    return resume_info

def seed_torch(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class LoggerProgressMeter(utils.ProgressMeter):
    """Extended ProgressMeter that also logs to file"""
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logging.info('\t'.join(entries))

def extractTopKColumns(matrix):
    '''
    Learn representative negative instances from each normal WSI
    '''
    score  = {}
    rank = np.linalg.matrix_rank(matrix)
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    
    for j in range(0, matrix.shape[1]):
        cscore = sum(np.square(vh[0:rank,j]))
        cscore /= rank
        score[j] = min(1, rank*cscore)
        
    prominentColumns = sorted(score, key=score.get, reverse=True)[:rank]
    #Removal of extra dimension\n",
    C = np.squeeze(matrix[:, [prominentColumns]])
    
    return ({"columns": prominentColumns, "matrix": C, "scores": sorted(score.values(), reverse = True)[:rank]})
def compute_regression_metrics(y_true, y_pred):
    """
    y_true, y_pred: 1D torch tensors
    Returns:
      pearson_r, pearson_p,
      spearman_r, spearman_p,
      mse, mae, r2
    """
    y_true_np = y_true.detach().cpu().numpy().astype(float)
    y_pred_np = y_pred.detach().cpu().numpy().astype(float)

    # Handle constant arrays safely for correlations
    if np.all(y_true_np == y_true_np[0]) or np.all(y_pred_np == y_pred_np[0]):
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan
    else:
        pearson_r, pearson_p = pearsonr(y_true_np, y_pred_np)
        spearman_r, spearman_p = spearmanr(y_true_np, y_pred_np)

    mse = mean_squared_error(y_true_np, y_pred_np)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    return pearson_r, pearson_p, spearman_r, spearman_p, mse, mae, r2

def run(args):
    # List all .npy files in the folder
    npy_files = [f for f in os.listdir(args.embd_dir) if f.endswith(".npy")]
    
    # Store best metrics across all folds
    best_fold_metrics = []
    
    # Run training for each fold
    for fold in range(args.nfold):
        if not fold==args.fold_idx:
            continue

        fold_header = f"\n{'='*20} Fold {fold + 1}/{args.nfold} {'='*20}"
        print(fold_header)
        logging.info(fold_header)

        # Check for existing checkpoint if resume is requested
        checkpoint_dir = os.path.join(args.save, f"fold_{fold}")
        resume_info = None
        
        if args.resume:
            checkpoint_path, latest_epoch = find_latest_checkpoint(checkpoint_dir)
            if checkpoint_path:
                logging.info(f"Found checkpoint: {checkpoint_path} at epoch {latest_epoch}")
            else:
                logging.info("No checkpoint found, starting from scratch")

        wsi_dataset_train = pd.read_csv(f"{args.folds_dir}fold{fold}_train.csv")

        wsi_dataset_train["embeddings_filename"] = wsi_dataset_train["slide_filename"].apply(find_npy, args=(npy_files, args.embd_dir))
        wsi_dataset_train = wsi_dataset_train[wsi_dataset_train["embeddings_filename"].notna()]               # remove NaN
        wsi_dataset_train = wsi_dataset_train[wsi_dataset_train["embeddings_filename"] != ""]                 # remove empty string
        # wsi_dataset_train = wsi_dataset_train.sample(n=5)

        wsi_dataset_train = wsi_dataset_train.reset_index(drop=True)
        wsi_dataset_train = wsi_dataset_train[wsi_dataset_train['embeddings_filename'].apply(os.path.exists)].reset_index(drop=True)

        wsi_paths_train = wsi_dataset_train["embeddings_filename"]
        wsi_labels_train = wsi_dataset_train[f"{args.mutation}"]

        pos_count_train = (wsi_dataset_train[f"{args.mutation}"] == 1).sum()
        neg_count_train = (wsi_dataset_train[f"{args.mutation}"] == 0).sum()

        stats = f"\nSplit: train, Fold: {fold}\n"
        stats += f"Total slides: {len(wsi_dataset_train)}\n"
        stats += f"Positive slides: {pos_count_train}\n"
        stats += f"Negative slides: {neg_count_train}\n"
        print(stats)
        logging.info(stats)

        # Calculate pos_weight for BCEWithLogitsLoss
        # pos_weight = torch.tensor([args.pw * (neg_count_train / pos_count_train)], dtype=torch.float32).cuda()

        wsi_dataset_val = pd.read_csv(f"{args.folds_dir}fold{fold}_val.csv")

        wsi_dataset_val["embeddings_filename"] = wsi_dataset_val["slide_filename"].apply(find_npy, args=(npy_files, args.embd_dir))
        wsi_dataset_val = wsi_dataset_val[wsi_dataset_val["embeddings_filename"].notna()]               # remove NaN
        wsi_dataset_val = wsi_dataset_val[wsi_dataset_val["embeddings_filename"] != ""]                 # remove empty string
        # wsi_dataset_val = wsi_dataset_val.sample(n=1)

        wsi_dataset_val = wsi_dataset_val.reset_index(drop=True)
        wsi_dataset_val = wsi_dataset_val[wsi_dataset_val['embeddings_filename'].apply(os.path.exists)].reset_index(drop=True)

        wsi_paths_val = wsi_dataset_val["embeddings_filename"]
        wsi_labels_val = wsi_dataset_val[f"{args.mutation}"]

        pos_count_val = (wsi_dataset_val[f"{args.mutation}"] == 1).sum()
        neg_count_val = (wsi_dataset_val[f"{args.mutation}"] == 0).sum()

        stats = f"\nSplit: val, Fold: {fold}\n"
        stats += f"Total slides: {len(wsi_dataset_val)}\n"
        stats += f"Positive slides: {pos_count_val}\n"
        stats += f"Negative slides: {neg_count_val}\n"
        print(stats)
        logging.info(stats)

        wsi_dataset_test = pd.read_csv(f"{args.folds_dir}fold{fold}_test.csv")

        wsi_dataset_test["embeddings_filename"] = wsi_dataset_test["slide_filename"].apply(find_npy, args=(npy_files, args.embd_dir))
        wsi_dataset_test = wsi_dataset_test[wsi_dataset_test["embeddings_filename"].notna()]               # remove NaN
        wsi_dataset_test = wsi_dataset_test[wsi_dataset_test["embeddings_filename"] != ""]                 # remove empty string

        wsi_dataset_test = wsi_dataset_test.reset_index(drop=True)
        wsi_dataset_test = wsi_dataset_test[wsi_dataset_test['embeddings_filename'].apply(os.path.exists)].reset_index(drop=True)

        wsi_paths_test = wsi_dataset_test["embeddings_filename"]
        wsi_labels_test = wsi_dataset_test[f"{args.mutation}"]

        pos_count_test = (wsi_dataset_test[f"{args.mutation}"] == 1).sum()
        neg_count_test = (wsi_dataset_test[f"{args.mutation}"] == 0).sum()

        stats = f"\nSplit: test, Fold: {fold}\n"
        stats += f"Total slides: {len(wsi_dataset_test)}\n"
        stats += f"Positive slides: {pos_count_test}\n"
        stats += f"Negative slides: {neg_count_test}\n"
        print(stats)
        logging.info(stats)

        
        lowkeyset = np.ones((1, 4000, args.hidden_dim1))
        highkeyset = np.ones((1, 4000, args.hidden_dim1))

        
        # Initialize model, loss, and optimizer
        net = WSIClassifier(
            patch_dim=args.patch_dim,
            cell_dim=args.cell_dim,
            hidden_dim1=args.hidden_dim1,
            hidden_dim2=args.hidden_dim2,
            hidden_dim3=args.hidden_dim3,
            num_classes=args.num_classes,
            distance_metric=args.distance_metric,
            use_spatial_bias=args.use_spatial_bias,
            use_cell_ratios=args.use_cell_ratios,
            dropout=args.dropout,
            projection_dim=args.projection_dim,
            aggregation_method=args.aggregation_method,
            use_patch_embeddings=args.use_patch_embeddings,
            patch_embeddings_only=args.patch_embeddings_only,
            mil=args.mil,
            device=device,
            lowkeysetlength=lowkeyset.shape[1],
            highkeysetlength=highkeyset.shape[1]
        )
        criterions = [nn.MSELoss().to(device)]
        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        if args.regtype == "l2":
            optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.reg)
        else:
            optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0)
        
        net.cuda()
        
        # Load checkpoint if resuming
        start_epoch = 0
        if args.resume and checkpoint_path:
            resume_info = load_checkpoint(checkpoint_path, net, optimizer, device)
            start_epoch = resume_info['epoch']
        
        writer = SummaryWriter(os.path.join(args.save, f"fold_{fold}", 
                             time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())))
        
        # Set up early stopping
        EarlyStopping = utils.EarlyStopping(
            save_dir=os.path.join(args.save, f"fold_{fold}"), args=args)
        
        # Restore early stopping state if resuming
        if resume_info:
            EarlyStopping.counter = resume_info.get('early_stopping_counter', 0)
            EarlyStopping.best_score = resume_info.get('best_metric', 0 if args.monitor == 'acc' else 1 if args.monitor == 'auc' else float('inf'))
        
        monitor_values = {'loss': 2 }
        monitor_idx = monitor_values[args.monitor]

        # Training set
        train_dataset = WSIDataset(wsi_paths_train, wsi_labels_train, args.use_cell_ratios, args.use_single_cell, args.seed, args.nsamples, 'train')
        val_dataset = WSIDataset(wsi_paths_val, wsi_labels_val, args.use_cell_ratios, args.use_single_cell, args.seed, args.nsamples, 'val')
        test_dataset = WSIDataset(wsi_paths_test, wsi_labels_test, args.use_cell_ratios, args.use_single_cell, args.seed, args.nsamples, 'test')
        if args.weighted_sample:
            weights = train_dataset.get_weights()
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, 
                sampler=torch.utils.data.WeightedRandomSampler(weights, len(weights)),
                num_workers=args.num_workers, pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=False)

        # Validation set
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=False)

        # Test set
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=False)

        best_epoch = start_epoch
        for epoch in range(start_epoch, args.epochs):
            logging.info(f"\nStarting epoch {epoch}/{args.epochs}")

            # Train for one epoch
            lowkeyset_temp, highkeyset_temp = train(train_loader, net, criterions, optimizer, epoch, args, writer, lowkeyset, highkeyset, device)

            # Evaluate on validation set
            metrics = validate(val_loader, net, epoch, criterions, args, writer, 'val', lowkeyset, highkeyset, device )
            
            # Early stopping
            EarlyStopping(epoch, metrics[monitor_idx], net, optimizer)
            
            # Evaluate on testing set
            if EarlyStopping.early_stop:
                _ = validate(test_loader, net, epoch, criterions, args, writer, 'test', lowkeyset, highkeyset, device)
                early_stop_msg = f'****Early stop at epoch:{epoch-args.patience}'
                print(early_stop_msg)
                logging.info(early_stop_msg)
                break
            else:
                if EarlyStopping.counter == 0:
                    best_metrics = validate(test_loader, net, epoch, criterions, args, writer, 'test', lowkeyset, highkeyset, device)
                    best_epoch = epoch
                else:
                    _ = validate(test_loader, net, epoch, criterions, args, writer, 'test', lowkeyset, highkeyset, device)

            lowkeyset, highkeyset = lowkeyset_temp, highkeyset_temp
        
        best_fold_metrics.append(best_metrics)
        fold_result = (
            f'Fold {fold + 1} best testing result:\n'
            f'Epoch: {best_epoch}, '
            f'Pearson: {best_metrics[0]:.3f}, Spearman: {best_metrics[1]:.3f}, '
            f'MSE: {best_metrics[2]:.4f}, MAE: {best_metrics[3]:.4f}, R2: {best_metrics[4]:.3f}, '
            f'Pearson_p: {best_metrics[5]:.1e}, Spearman_p: {best_metrics[6]:.1e}'
        )
        print(fold_result)
        logging.info(fold_result)
    
    # Calculate and print average metrics across folds
    avg_metrics = np.mean(best_fold_metrics, axis=0)
    std_metrics = np.std(best_fold_metrics, axis=0)
    final_results = '\nAverage performance across folds:'
    final_results += f'\nPearson: {avg_metrics[0]:.3f}±{std_metrics[0]:.3f}'
    final_results += f'\nSpearman: {avg_metrics[1]:.3f}±{std_metrics[1]:.3f}'
    final_results += f'\nMSE: {avg_metrics[2]:.4f}±{std_metrics[2]:.4f}'
    final_results += f'\nMAE: {avg_metrics[3]:.4f}±{std_metrics[3]:.4f}'
    final_results += f'\nR2: {avg_metrics[4]:.3f}±{std_metrics[4]:.3f}'
    final_results += f'\nPearson_p: {avg_metrics[5]:.1e}±{std_metrics[5]:.1e}'
    final_results += f'\nSpearman_p: {avg_metrics[6]:.1e}±{std_metrics[6]:.1e}'
    print(final_results)
    logging.info(final_results)


def train(train_loader, model, criterions, optimizer, epoch, args, writer, lowkeyset, highkeyset, device):
    lowkey_tensor = torch.from_numpy(lowkeyset).float().to(device)
    highkey_tensor = torch.from_numpy(highkeyset).float().to(device)
    criterion = criterions[0]
    losses = utils.AverageMeter('Loss', ':.4e')
    progress = LoggerProgressMeter(len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch))
    model.train()
    all_outputs = []
    all_targets = []
    highkeys = np.empty((0, args.hidden_dim1))
    lowkeys = np.empty((0, args.hidden_dim1))
    for i, data in enumerate(train_loader):
        images, target = data
        target = target.to(device).float().view(-1)
        low = True if torch.all(target == 0) else False
        if args.mil == "casii":
            Y_hat, output, As, patch_embeddings = model((images, lowkey_tensor, highkey_tensor))
            output = output.view(-1).float()
        else:
            output, attention_weights, slide_embeddings, W_matrix, samples, patch_embeddings, cellpatchattn = model(images)
            output = output.view(-1).float()
        all_outputs.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())
        if args.mil == "casii":
            patch_embeddings_npy = patch_embeddings.squeeze(0).cpu().detach().numpy().T
            res = extractTopKColumns(patch_embeddings_npy)
            cols = res["columns"]
            keys = np.transpose(np.squeeze(patch_embeddings_npy[:, cols]))
            length = keys.shape[0]
            try:
                if length <= args.t:
                    if low:
                        lowkeys = np.vstack([lowkeys, keys])
                    else:
                        highkeys = np.vstack([highkeys, keys])
                else:
                    if low:
                        lowkeys = np.vstack([lowkeys, keys[:args.t]])
                    else:
                        highkeys = np.vstack([highkeys, keys[:args.t]])
            except:
                print(keys.shape)
                print(lowkeys.shape)
                print(highkeys.shape)
                print(keys[:10].shape)
        loss = criterion(output, target)
        if args.regtype == "l1" and args.reg > 0:
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += args.reg * l1_loss
        losses.update(loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.freq == 0:
            progress.display(i)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    pearson_r, pearson_p, spearman_r, spearman_p, mse, mae, r2 = compute_regression_metrics(all_targets, all_outputs)
    train_metrics = (f'Training metrics - Epoch [{epoch}] '
                     f'Pearson: {pearson_r:.3f} (p={pearson_p:.1e}), '
                     f'Spearman: {spearman_r:.3f} (p={spearman_p:.1e}), '
                     f'MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.3f}, '
                     f'Loss(MSE): {losses.avg:.4f}')
    logging.info(train_metrics)
    if writer:
        writer.add_scalar("Loss/train", losses.avg, epoch)
        writer.add_scalar("MSE/train", mse, epoch)
        writer.add_scalar("MAE/train", mae, epoch)
        writer.add_scalar("R2/train", r2, epoch)
        if not np.isnan(pearson_r):
            writer.add_scalar("Pearson_r/train", pearson_r, epoch)
        if not np.isnan(spearman_r):
            writer.add_scalar("Spearman_r/train", spearman_r, epoch)
    return lowkeys, highkeys





def validate(val_loader, model, epoch, criterions, args, writer, val='val', lowkeyset=None, highkeyset=None, device=None):
    lowkey_tensor = torch.from_numpy(lowkeyset).float().to(device)
    highkey_tensor = torch.from_numpy(highkeyset).float().to(device)
    criterion = criterions[0]
    losses = utils.AverageMeter('Loss', ':.4e')
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data
            target = target.to(device).float().view(-1)
            if args.mil == "casii":
                Y_hat, output, As, patch_embeddings = model((images, lowkey_tensor, highkey_tensor))
            else:
                output, attention_weights, slide_embeddings, W_matrix, samples, patch_embeddings, cellpatchattn = model(images)
            output = output.view(-1).float()
            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())
            loss = criterion(output, target)
            losses.update(loss.item(), args.batch_size)
            if device.type == "cuda":
                torch.cuda.synchronize()
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                mem_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"[Slide {i+1}/{len(val_loader)}] CUDA Memory — Allocated: {mem_alloc:.2f} MB | Reserved: {mem_reserved:.2f} MB | Max: {mem_max:.2f} MB")
                if args.mil != "casii":
                    del output, attention_weights, slide_embeddings, W_matrix, samples, patch_embeddings, cellpatchattn
                else:
                    del output, Y_hat, As, patch_embeddings
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                mem_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"[Slide {i+1}/{len(val_loader)}] CUDA Memory (post-GC) — Allocated: {mem_alloc:.2f} MB | Reserved: {mem_reserved:.2f} MB | Max: {mem_max:.2f} MB")
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    pearson_r, pearson_p, spearman_r, spearman_p, mse, mae, r2 = compute_regression_metrics(all_targets, all_outputs)
    if val == 'val':
        metrics_msg = (f' **Validation '
                       f'Pearson {pearson_r:.3f} (p={pearson_p:.1e}) '
                       f'Spearman {spearman_r:.3f} (p={spearman_p:.1e}) '
                       f'MSE {mse:.4f} MAE {mae:.4f} R2 {r2:.3f} '
                       f'Loss(MSE) {losses.avg:.4f}')
    else:
        metrics_msg = (f' ***Testing '
                       f'Pearson {pearson_r:.3f} (p={pearson_p:.1e}) '
                       f'Spearman {spearman_r:.3f} (p={spearman_p:.1e}) '
                       f'MSE {mse:.4f} MAE {mae:.4f} R2 {r2:.3f} '
                       f'Loss(MSE) {losses.avg:.4f}')
    print(metrics_msg)
    logging.info(metrics_msg)
    if writer:
        writer.add_scalar("Loss/"+val, losses.avg, epoch)
        writer.add_scalar("MSE/"+val, mse, epoch)
        writer.add_scalar("MAE/"+val, mae, epoch)
        writer.add_scalar("R2/"+val, r2, epoch)
        if not np.isnan(pearson_r):
            writer.add_scalar("Pearson_r/"+val, pearson_r, epoch)
        if not np.isnan(spearman_r):
            writer.add_scalar("Spearman_r/"+val, spearman_r, epoch)
    return pearson_r, spearman_r, mse, mae, r2, pearson_p, spearman_p




if __name__ == '__main__':
    args = parser.parse_args()
    
    save_dir = "/fs/scratch/PAS2942/Alejandro/RT/{}/runs/{}_s{}".format(
        args.code, args.code, args.seed)
    folds_dir = f"/fs/scratch/PAS2942/Alejandro/RT/folds/"
    
    embd_dir = f"/fs/scratch/PAS2942/Users/AbdulRehman/ResearchProjects/genemutations/{args.organ}_dataset/"

    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)    
    
    seed_torch(args.seed)
    args.save = save_dir
    args.folds_dir = folds_dir
    args.embd_dir = embd_dir

    # Setup logging before starting the run
    log_file = setup_logging(args)
    logging.info(f"Log file created at: {log_file}")

    run(args)
