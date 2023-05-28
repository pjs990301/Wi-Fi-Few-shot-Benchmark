from dataset import *
from UT_HAR_model import *
from ReWiS_model import *
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_UT_HAR_data(root):
    print('using dataset: UT-HAR DATA')
    data = UT_HAR_dataset(root)
    train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
    test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'], data['X_test']), 0),
                                              torch.cat((data['y_val'], data['y_test']), 0))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)  # drop_last=True
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, test_loader

def load_ReWiS_data(root):
    print('using dataset: ReWiS DATA')
    train_x, train_y = read_csi(root + '/few_shot_datasets/ReWis/m1c4_PCA_test_80/train_A1')
    train_x = np.expand_dims(train_x, axis=1)
    test_x, test_y = read_csi(root + '/few_shot_datasets/ReWis/m1c4_PCA_test_80/test_A3')
    test_x = np.expand_dims(test_x, axis=1)

    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)
    test_y = label_encoder.transform(test_y) 

    train_x, train_y = torch.tensor(train_x).float(), torch.tensor(train_y)
    test_x, test_y = torch.tensor(test_x).float(), torch.tensor(test_y)

    train_set = torch.utils.data.TensorDataset(train_x, train_y)
    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, test_loader

def load_ReWiS_data_split(root):
    print('using dataset: ReWiS DATA Split')
    X, y = read_csi(root + '/few_shot_datasets/ReWis/m1c4_PCA_test_80/train_A1')
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    train_x = np.expand_dims(train_x, axis=1)
    test_x = np.expand_dims(test_x, axis=1)
    label_encoder = LabelEncoder()

    train_y = label_encoder.fit_transform(train_y)
    test_y = label_encoder.transform(test_y) 

    train_x, train_y = torch.tensor(train_x).float(), torch.tensor(train_y)
    test_x, test_y = torch.tensor(test_x).float(), torch.tensor(test_y)

    train_set = torch.utils.data.TensorDataset(train_x, train_y)
    test_set = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, test_loader

def load_UT_HAR_supervised_model(model_name):
    if model_name == 'LeNet':
        print("using model: LeNet")
        model = UT_HAR_LeNet()
        train_epoch = 100  # 40

    elif model_name == 'ResNet50':
        print("using model: ResNet50")
        model = UT_HAR_ResNet50()
        train_epoch = 100  # 100

    elif model_name == 'RNN':
        print("using model: RNN")
        model = UT_HAR_RNN()
        train_epoch = 100  # 20

    elif model_name == 'LSTM':
        print("using model: LSTM")
        model = UT_HAR_LSTM()
        train_epoch = 100

    elif model_name == 'BiLSTM':
        print("using model: BiLSTM")
        model = UT_HAR_BiLSTM()
        train_epoch = 100

    return model, train_epoch

def load_ReWiS_supervised_model(model_name):
    if model_name == 'LeNet':
        print("using model: LeNet")
        model = ReWiS_LeNet()
        train_epoch = 100  # 40

    elif model_name == 'ResNet50':
        print("using model: ResNet50")
        model = ReWiS_ResNet50()
        train_epoch = 100  # 100

    elif model_name == 'RNN':
        print("using model: RNN")
        model = ReWiS_RNN()
        train_epoch = 100  # 20

    elif model_name == 'LSTM':
        print("using model: LSTM")
        model = ReWiS_LSTM()
        train_epoch = 100

    elif model_name == 'BiLSTM':
        print("using model: BiLSTM")
        model = ReWiS_BiLSTM()
        train_epoch = 100

    return model, train_epoch