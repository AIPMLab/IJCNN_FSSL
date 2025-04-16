import argparse
import os
import torch

import numpy as np
import torch
import random



def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    else:
        print("Non-deterministic")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    # training specific args
    parser.add_argument('--dataset', type=str, default='medical', help='choose from random, stl10, mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_from', type=str, default=None)

    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--use_default_hyperparameters', action='store_true')
    # model related params
    parser.add_argument('--model', type=str, default='byol')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_epochs', type=int, default=100, help='This will affect learning rate decay')
    parser.add_argument('--stop_at_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--proj_layers', type=int, default=None, help="number of projector layers. In cifar experiment, this is set to 2")
    # optimization params
    parser.add_argument('--optimizer', type=str, default='lars_simclr', help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--warmup_lr', type=float, default=0, help='Initial warmup learning rate')
    parser.add_argument('--base_lr', type=float, default=0.3)
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--initial-lr', default=0.0, type=float,metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',help='length of learning rate cosine rampdown (>= length of training)')
 
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1.5e-6)

    parser.add_argument('--eval_after_train', type=str, default=None)
    parser.add_argument('--head_tail_accuracy', action='store_true', help='the acc in first epoch will indicate whether collapse or not, the last epoch shows the final accuracy')
    
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: E")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")

    parser.add_argument('--label_rate', type=float, default=0.1, help="the fraction of labeled data")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--threshold_pl', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--phi_g', type=int, default=10, help="tipping point 1")
    parser.add_argument('--psi_g', type=int, default=40, help="tipping point 2")
    parser.add_argument('--comu_rate',type=float, default=0.5,help="the comu_rate of ema model")
    parser.add_argument('--ramp',type=str,default='linear', help="ramp of comu")
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--T', type=float, default=2.0, help='Temperature parameter for sharpening pseudo labels.')
    
    parser.add_argument('--iid', type=str, default='iid', help='iid')
    args = parser.parse_args()
    
    if args.debug:
        args.batch_size = 2 
        args.stop_at_epoch = 2
        args.num_epochs = 3 # train only one epoch
        args.num_workers = 0

    assert not None in [args.output_dir, args.data_dir]
    os.makedirs(args.output_dir, exist_ok=True)
    # assert args.stop_at_epoch <= args.num_epochs
    if args.stop_at_epoch is not None:
        if args.stop_at_epoch > args.num_epochs:
            raise Exception
    else:
        args.stop_at_epoch = args.num_epochs

    if args.use_default_hyperparameters:
        raise NotImplementedError
    return args
