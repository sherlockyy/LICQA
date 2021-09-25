import os
import numpy as np
import random
from pathlib import Path
import json
import torch

from params import args
import network
from dataset import Data
from loss import Loss
from trainer import Trainer
from evaluator import Evaluator


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(args.seed)
random.seed(args.seed)


def main():
    if not args.test_only:
        # Split database, train and valid model
        Path(args.log_root).mkdir(parents=True, exist_ok=True)
        with open(Path(args.log_root)/'args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        args.device = torch.device("cuda" if torch.cuda.is_available() and ~args.cpu else "cpu")

        args.log_dir = Path(args.log_root)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

        args.data_train = [Path(args.train_files) / f'exp{args.exp_id}' / 'train.csv']
        args.data_val = [Path(args.train_files) / f'exp{args.exp_id}' / 'val.csv']
        args.data_test = [Path(args.train_files) / f'exp{args.exp_id}' / 'test.csv']

        _model = network.Model(args)
        _loader = Data(args)
        _loss = Loss(args) if not args.test_only else None
        t = Trainer(args, _model, _loader, _loss)
        _best_info = t.main_worker()

        print('\nBest info:\n')
        print(_best_info)
    
    else:
        # Predict on validation set
        args.device = torch.device("cuda" if torch.cuda.is_available() and ~args.cpu else "cpu")
        if not args.pre_train:
            raise ValueError("A pre-trained model is needed!")
        t = Evaluator(args)
        t.predict()


if __name__ == '__main__':
    main()
