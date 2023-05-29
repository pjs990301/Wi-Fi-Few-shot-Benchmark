from dataset import *
from UT_HAR_model import *
from ReWiS_model import *
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def load_UT_HAR_data(root):
    print('using dataset: UT-HAR DATA')
    data = UT_HAR_dataset(root)
    train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
    test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'], data['X_test']), 0),
                                              torch.cat((data['y_val'], data['y_test']), 0))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)  # drop_last=True
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, test_loader

def load_ReWiS_data_fewshot(root) :
    print('using dataset: ReWiS DATA Few shot')
    train_x, train_y = read_csi(root + '/few_shot_datasets/ReWis/m1c4_PCA_test_80/train_A1')
    train_x = np.expand_dims(train_x, axis=1)
    test_x, test_y = read_csi(root + '/few_shot_datasets/ReWis/m1c4_PCA_test_80/test_A3')
    # test_x, test_y = read_csi_csv(root + '/few_shot_datasets/test',one_file=True)
    test_x = np.expand_dims(test_x, axis=1)

    return train_x, train_y, test_x, test_y

def load_UT_HAR_supervised_model(model_name):
    if model_name == 'LeNet':
        print("using model: LeNet")
        model = UT_HAR_LeNet()

    elif model_name == 'ResNet50':
        print("using model: ResNet50")
        model = UT_HAR_ResNet50()

    elif model_name == 'RNN':
        print("using model: RNN")
        model = UT_HAR_RNN()

    elif model_name == 'LSTM':
        print("using model: LSTM")
        model = UT_HAR_LSTM()

    elif model_name == 'BiLSTM':
        print("using model: BiLSTM")
        model = UT_HAR_BiLSTM()

    elif model_name == 'ViT':
        print("using model: ViT")
        model = UT_HAR_ViT()

    return model

def load_ReWiS_supervised_model(model_name):
    if model_name == 'LeNet':
        print("using model: LeNet")
        model = ReWiS_LeNet()

    elif model_name == 'ResNet50':
        print("using model: ResNet50")
        model = ReWiS_ResNet50()

    elif model_name == 'RNN':
        print("using model: RNN")
        model = ReWiS_RNN()

    elif model_name == 'LSTM':
        print("using model: LSTM")
        model = ReWiS_LSTM()

    elif model_name == 'BiLSTM':
        print("using model: BiLSTM")
        model = ReWiS_BiLSTM()

    elif model_name == 'ViT':
        print("using model: ViT")
        model = ReWiS_ViT(
            in_channels=1,  # 입력 채널 수
            patch_size=[22, 22],  # 패치 크기 (가로, 세로)
            embed_dim=64,  # 임베딩 차원
            num_layers=12,  # Transformer 블록 수
            num_heads=8,  # 멀티헤드 어텐션에서의 헤드 수
            mlp_dim=4,  # MLP의 확장 비율
            num_classes=4,  # 분류할 클래스 수
            in_size=[242, 242]  # 입력 이미지 크기 (가로, 세로)
        )

    return model


def euclidean_dist(x, y):
    """
    Computes euclidean distance btw x and y
    Args:
        x (torch.Tensor): shape (n, d). n usually n_way*n_query
        y (torch.Tensor): shape (m, d). m usually n_way
    Returns:
        torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def extract_train_sample(n_way, n_support, n_query, datax, datay):

    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): dataset of dataloader dataframes
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
          (torch.Tensor): sample of dataloader dataframes. Size (n_way, n_support+n_query, (dim))
          (int): n_way
          (int): n_support
          (int): n_query
    """
    sample = None
    K = np.random.choice(np.unique(datay), n_way, replace=False)

    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        if sample is None:
            sample = np.array([sample_cls])
        else:
            sample = np.vstack([sample, [np.array(sample_cls)]])
        #sample.append(sample_cls)

    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()

    # sample = sample.permute(0,1,4,2,3)
    # sample = np.expand_dims(sample, axis= 0)
    return ({
        'csi_mats': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


def extract_test_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support+n_querry, for n_way classes
    Args:
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        datax (np.array): dataset of csi dataframes
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
          (torch.Tensor): sample of csi dataframes. Size (n_way, n_support+n_query, (dim))
          (int): n_way
          (int): n_support
          (int): n_query
    """
    #K = np.array(['empty', 'jump', 'stand', 'walk']) # ReWis
    K = np.array(param['test_labels'])

    # extract support set & query set
    support_sample = []
    query_sample = []
    for cls in K:
        datax_cls = datax[datay == cls]
        # print(datax_cls.shape)
        # print(datax_cls.dtype)

        support_cls = datax_cls[:n_support]
        query_cls = np.array(datax_cls[n_support:n_support+n_query])

        # print(query_cls.shape)
        # print(query_cls.dtype)
        # print("---------")

        support_sample.append(support_cls)
        query_sample.append(query_cls)
    
    support_sample = np.array(support_sample)
    query_sample = np.array(query_sample)

    # print(support_sample.dtype)
    # print(type(support_sample))

    # print(query_sample.dtype)
    # print(type(query_sample))

    support_sample = torch.from_numpy(support_sample).float()
    query_sample = torch.from_numpy(query_sample).float()

    return ({
        's_csi_mats': support_sample,
        'q_csi_mats': query_sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })

