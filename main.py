import torch.nn as nn
import argparse
from util import load_UT_HAR_data, load_ReWiS_data, load_ReWiS_data_split, load_ReWiS_data_fewshot, load_Home_data_fewshot
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
    parser.add_argument('--dataset', choices=['UT_HAR', 'ReWiS', 'Home'])
    parser.add_argument('--model',   choices=['LeNet', 'ResNet50',  'RNN', 'LSTM', 'BiLSTM','ViT'])
    parser.add_argument('--learning', choices=['supervised', 'few-shot'], required=True)
    parser.add_argument('--split', default='F', choices=['T','F'])
    parser.add_argument('--epoch', default=100)
    # parser.add_argument('--MHz', default='80MHz')
    # parser.add_argument('--train_env', default='A1_1_4')
    # parser.add_argument('--test_env', default='A3_1_4')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_epoch = int(args.epoch)
    criterion = nn.CrossEntropyLoss()

    # train_dir = args.train_env
    # test_dir = args.test_env
    # MHz = args.MHz

    if args.learning == 'supervised':
        # if args.dataset == 'UT_HAR':
        #     train_loader, test_loader = load_UT_HAR_data(root)
        #     model = load_UT_HAR_supervised_model(args.model)

        if args.dataset == 'ReWiS':
            if args.split == 'T' :
                # train_loader, test_loader = load_ReWiS_data_split(root, MHz, train_dir)
                train_loader, test_loader = load_ReWiS_data_split(root)

            else :    
                # train_loader, test_loader = load_ReWiS_data(root, MHz, train_dir, test_dir)
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
        test_acc, test_loss, test_accuracy_history, test_loss_history= supervised.test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion,
            device=device
        )
    
        # model_out = f'Result/{args.learning}/{args.dataset}/{MHz}_{train_dir}_{test_dir}_{args.split}/{args.model}/{args.epoch}_{test_acc:.3f}/'
        model_out = 'Result/{}/{}_{}/{}/{}_{:.3f}/'.format(args.learning, args.dataset, args.split, args.model, args.epoch, test_acc)
        if not os.path.exists(model_out):
            os.makedirs(model_out)

        # Result/learning/dataset/train_env_test_env/split/model_name/epoch/train.csv
        # Result/learning/dataset/split/model_name/epoch/test.csv

        train_history = pd.DataFrame({'Epoch': range(0, train_epoch),
                       'Accuracy': train_accuracy_history,
                       'Loss': train_loss_history})
        
        test_history = pd.DataFrame({'Test Accuracy': [test_acc],
                        'Test Loss': [test_loss]})
        
        test_history2 = pd.DataFrame({'Test Accuracy': [test_accuracy_history],
                        'Test Loss': [test_loss_history]})
        

        
        train_history.to_csv(model_out + 'train.csv', index=False)    
        test_history.to_csv(model_out + 'test.csv', index=False)
        test_history2.to_csv(model_out + 'test_list.csv', index=False)   
   
        torch.save(model.state_dict(), model_out + 'model.pt')
 
    elif args.learning == 'few-shot':
        if args.dataset == 'ReWiS' :
            train_x, train_y, test_x, test_y = load_ReWiS_data_fewshot(root)
            if args.model == 'ViT' :
                model = proto.load_protonet_vit(
                    # in_channels=1,  # 입력 채널 수
                    # patch_size=[16, 64],  # 패치 크기 (세로, 가로) 242 = 2 * 11 * 11
                    # embed_dim=64,  # 임베딩 차원
                    # num_layers=12,  # Transformer 블록 수
                    # num_heads=8,  # 멀티헤드 어텐션에서의 헤드 수
                    # mlp_dim=4,  # MLP의 확장 비율
                    # num_classes=4,  # 분류할 클래스 수
                    # in_size=[64, 64]  # 입력 이미지 크기 (가로, 세로)
                    
                    in_channels=1,  # 입력 채널 수
                    patch_size=[22, 242],  # 패치 크기 (세로, 가로) 242 = 2 * 11 * 11
                    embed_dim=64,  # 임베딩 차원
                    num_layers=12,  # Transformer 블록 수
                    num_heads=8,  # 멀티헤드 어텐션에서의 헤드 수
                    mlp_dim=4,  # MLP의 확장 비율
                    num_classes=4,  # 분류할 클래스 수
                    in_size=[242, 242]  # 입력 이미지 크기 (가로, 세로)
                )
            else :   
                args.model = 'ProtoNet'
                model = proto.load_protonet_conv(
                    x_dim=(1, 242, 242),
                    hid_dim=64,
                    z_dim=64,
                )
        elif args.dataset == 'Home' :
            train_x, train_y, test_x, test_y = load_Home_data_fewshot(root)
            
            if args.model == 'ViT' :
                model = proto.load_protonet_vit(
                    # in_channels=1,  # 입력 채널 수
                    # patch_size=[16, 64],  # 패치 크기 (세로, 가로) 
                    # embed_dim=64,  # 임베딩 차원
                    # num_layers=12,  # Transformer 블록 수
                    # num_heads=8,  # 멀티헤드 어텐션에서의 헤드 수
                    # mlp_dim=4,  # MLP의 확장 비율
                    # num_classes=4,  # 분류할 클래스 수
                    # in_size=[64, 64]  # 입력 이미지 크기 (가로, 세로)
                    
                    in_channels=1,  # 입력 채널 수
                    patch_size=[22, 242],  # 패치 크기 (세로, 가로) 242 = 2 * 11 * 11
                    embed_dim=64,  # 임베딩 차원
                    num_layers=12,  # Transformer 블록 수
                    num_heads=8,  # 멀티헤드 어텐션에서의 헤드 수
                    mlp_dim=4,  # MLP의 확장 비율
                    num_classes=4,  # 분류할 클래스 수
                    in_size=[242, 242]  # 입력 이미지 크기 (가로, 세로)
            )
        
            
        print("train_way: " + str(param['train_way']))
        print("train_support : " + str(param['train_support']))
        print("train_query : " + str(param['train_query']))
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

        print("test_way: " + str(param['test_way']))
        print("test_support : " + str(param['test_support']))
        print("test_query : " + str(param['test_query']))

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
        # Result/learning/dataset/model_name/max_epoch/epoch_size/conf.csv
        # model_out = f'Result/{args.learning}/{args.dataset}/{MHz}_{train_dir}_{test_dir}/{args.model}/train_s{param['train_support']}_train_q{param['train_query']}_test_s{param['test_support']}_test_q{param['test_query']}/{param['max_epoch']}_{param['epoch_size']}_{test_acc:.3f}/'
        model_out = (
                f"Result/{args.learning}/{args.dataset}/{args.model}/"
                f"train_s{param['train_support']}_train_q{param['train_query']}_"
                f"test_s{param['test_support']}_test_q{param['test_query']}/"
                f"{param['max_epoch']}_{param['epoch_size']}_{test_acc:.3f}/"
                )
        print(model_out)
        if not os.path.exists(model_out):
            os.makedirs(model_out)
        # model_out = 'Result/{}/{}/{}/{}/{}/train_s{}_train_q{}_test_s{}_test_q{}/{}/'.format(args.learning, args.dataset, args.model, param['max_epoch'], param['epoch_size'],param['train_support'],param['train_query'],param['test_support'],param['test_query'],test_acc)
        
        
        train_history = pd.DataFrame({'Epoch': range(0, param['max_epoch']),
                       'Accuracy': train_accuracy_history,
                       'Loss': train_loss_history})
        test_history = pd.DataFrame({'Test Accuracy': [test_acc]})
        confusion_matrix = pd.DataFrame(conf_mat.numpy())
        
        train_history.to_csv(model_out+ 'train.csv', index=False)
        test_history.to_csv(model_out+ 'test.csv',index=False)
        confusion_matrix.to_csv(model_out+'confusion.csv', index=True)    
        torch.save(model.state_dict(),model_out + 'model.pt')

if __name__ == "__main__":
    main()
