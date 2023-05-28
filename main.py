import torch.nn as nn
import argparse
from util import load_UT_HAR_data, load_ReWiS_data, load_ReWiS_data_split
from util import load_UT_HAR_supervised_model, load_ReWiS_supervised_model
import torch
import numpy as np
import supervised
import torch.backends.cudnn as cudnn
import random


''' 
fix seed
'''
torch.manual_seed(0)
torch.cuda.manual_seed(0)   
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)


def main():
    root = './Data'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['UT_HAR_data', 'ReWiS'])
    parser.add_argument('--model',
                        choices=['LeNet', 'ResNet50',  'RNN', 'LSTM', 'BiLSTM' ])
    parser.add_argument('--learning', choices=['supervised', 'few-shot'])
    parser.add_argument('--dataset_split', default='F', choices=['T','F'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    if args.learning == 'supervised':
        if args.dataset == 'UT_HAR_data':
            train_loader, test_loader = load_UT_HAR_data(root)
            model, train_epoch = load_UT_HAR_supervised_model(args.model)
            supervised.train(
                model=model,
                tensor_loader=train_loader,
                num_epochs=train_epoch,
                learning_rate=1e-3,
                criterion=criterion,
                device=device
            )
            supervised.test(
                model=model,
                tensor_loader=test_loader,
                criterion=criterion,
                device=device
            )

        if args.dataset == 'ReWiS':
            if args.dataset_split == 'T' :
                train_loader, test_loader = load_ReWiS_data_split(root)

            else :    
                train_loader, test_loader = load_ReWiS_data(root)
            model, train_epoch = load_ReWiS_supervised_model(args.model)
            supervised.train(
                model=model,
                tensor_loader=train_loader,
                num_epochs=train_epoch,
                learning_rate=1e-3,
                criterion=criterion,
                device=device
            )
            supervised.test(
                model=model,
                tensor_loader=test_loader,
                criterion=criterion,
                device=device
            )

    elif args.learning == 'few-shot':
        print()


if __name__ == "__main__":
    main()
