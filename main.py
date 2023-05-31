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
import os
import pandas as pd

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

        train_accuracy_history, train_loss_history= supervised.train(
            model=model,
            tensor_loader=train_loader,
            num_epochs=train_epoch,
            learning_rate=1e-3,
            criterion=criterion,
            device=device
        )
        test_acc, test_loss = supervised.test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion,
            device=device
        )
  
        os.makedirs('Result/{}/{}/{}/{}/{}/'.format(args.learning, args.dataset, args.split, args.model, args.epoch))

        # Result/learning/dataset/split/model_name/epoch/train.csv
        # Result/learning/dataset/split/model_name/epoch/test.csv

        train_history = pd.DataFrame({'Epoch': range(0, train_epoch),
                       'Accuracy': train_accuracy_history,
                       'Loss': train_loss_history})
        test_history = pd.DataFrame({'Test Accuracy': [test_acc],
                        'Test Loss': [test_loss]})

        train_history.to_csv('Result/{}/{}/{}/{}/{}/train.csv'.format(args.learning, args.dataset, args.split, args.model, args.epoch), index=False)    
        test_history.to_csv('Result/{}/{}/{}/{}/{}/test.csv'.format(args.learning, args.dataset, args.split, args.model, args.epoch), index=False)   
        torch.save(model.state_dict(),'Result/{}/{}/{}/{}/{}/model.pt'.format(args.learning, args.dataset, args.split, args.model, args.epoch))
 
    elif args.learning == 'few-shot':
        train_x, train_y, test_x, test_y = load_ReWiS_data_fewshot(root)

        if args.model == 'ViT' :
            model = proto.load_protonet_vit()
        else :   
            args.model = 'ProtoNet'
            model = proto.load_protonet_conv(
                x_dim=(1, 242, 242),
                hid_dim=64,
                z_dim=64,
            )
            

        train_accuracy_history, train_loss_history = few_shot.train(
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

        conf_mat, test_acc = few_shot.test(
            model = model,
            test_x = test_x,
            test_y = test_y,
            n_way = param['test_way'],
            n_support = param['test_support'],
            n_query = param['test_query'],
            test_episode = 1,
            device = device
        )

        # Result/learning/dataset/model_name/max_epoch/epoch_size/train.csv
        # Result/learning/dataset/model_name/max_epoch/epoch_size/test.csv
        # Result/learning/dataset//model_name/max_epoch/epoch_size/conf.csv
        os.makedirs('Result/{}/{}/{}/{}/{}/'.format(args.learning, args.dataset, args.model, param['max_epoch'], param['epoch_size']))
        
        train_history = pd.DataFrame({'Epoch': range(0, param['max_epoch']),
                       'Accuracy': train_accuracy_history,
                       'Loss': train_loss_history})
        test_history = pd.DataFrame({'Test Accuracy': [test_acc]})
        confusion_matrix = pd.DataFrame(conf_mat.numpy())
        
        train_history.to_csv('Result/{}/{}/{}/{}/{}/train.csv'.format(args.learning, args.dataset, args.model, param['max_epoch'], param['epoch_size']), index=False)
        test_history.to_csv('Result/{}/{}/{}/{}/{}/test.csv'.format(args.learning, args.dataset, args.model, param['max_epoch'], param['epoch_size']), index=False)    
        confusion_matrix.to_csv('Result/{}/{}/{}/{}/{}/confusion.csv'.format(args.learning, args.dataset, args.model, param['max_epoch'], param['epoch_size']), index=True)    
        torch.save(model.state_dict(),'Result/{}/{}/{}/{}/{}/model.pt'.format(args.learning, args.dataset, args.model, param['max_epoch'], param['epoch_size']))

if __name__ == "__main__":
    main()
