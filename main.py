import torch.nn as nn
import argparse
from util import load_UT_HAR_data, load_ReWiS_data, load_ReWiS_data_split, load_ReWiS_data_fewshot
from util import load_UT_HAR_supervised_model, load_ReWiS_supervised_model
import few_shot
import torch
import numpy as np
import supervised
import torch.backends.cudnn as cudnn
import random
import proto
from config import param
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
    parser.add_argument('--dataset', choices=['UT_HAR', 'ReWiS'])
    parser.add_argument('--model',   choices=['LeNet', 'ResNet50',  'RNN', 'LSTM', 'BiLSTM','ViT'])
    parser.add_argument('--learning', choices=['supervised', 'few-shot'], required=True)
    parser.add_argument('--split', default='F', choices=['T','F'])
    parser.add_argument('--epoch', default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_epoch = int(args.epoch)
    criterion = nn.CrossEntropyLoss()

    if args.learning == 'supervised':
        if args.dataset == 'UT_HAR':
            train_loader, test_loader = load_UT_HAR_data(root)
            model = load_UT_HAR_supervised_model(args.model)

        elif args.dataset == 'ReWiS':
            if args.split == 'T' :
                train_loader, test_loader = load_ReWiS_data_split(root)
            else :    
                train_loader, test_loader = load_ReWiS_data(root)
            model = load_ReWiS_supervised_model(args.model)

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
        train_x, train_y, test_x, test_y = load_ReWiS_data_fewshot(root)

        if args.model == 'ViT' :
            model = proto.load_protonet_vit()
        else :   
            model = proto.load_protonet_conv(
                x_dim=(1, 242, 242),
                hid_dim=64,
                z_dim=64,
            )

        few_shot.train(
            model = model, 
            learning_rate=1e-3, 
            train_x = train_x, 
            train_y = train_y,
            n_way = param['train_way'],
            n_support = param['train_support'], 
            n_query = param['train_query'],
            max_epoch = param['max_epoch'],
            epoch_size = param['epoch_size'],
            device = device
        )
        few_shot.test(
            model = model,
            test_x = test_x,
            test_y = test_y,
            n_way = param['test_way'],
            n_support = param['test_support'],
            n_query = param['test_query'],
            test_episode = 1,
            device = device
        )

if __name__ == "__main__":
    main()
