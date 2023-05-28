import numpy as np
from glob import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing as mp
from config import param
from scipy.io import loadmat
from dataloader.pcapTodf import pcap_to_df
import pandas as pd

def UT_HAR_dataset(root_dir):
    data_list = glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI Data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob(root_dir+'/*/*.mat')
        self.folder = glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x, y

def read_mat(csi_directory_path, csi_action):
    """
    Reads all the actions from a given activity directory
    """
    datax = []
    datay = []

    csi_mats = os.listdir(csi_directory_path)
    for csi_mat in csi_mats:
        mat = loadmat(csi_directory_path + csi_mat)
        if 'PCA' in csi_directory_path:
            data = mat['cfm_data']
        else:
            data = mat['iq_data']

        datax.extend([data])
        datay.extend([csi_action])
    return np.array(datax), np.array(datay)


def read_csi(base_directory):
    """
    Reads all the data_frames from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(read_mat, args=(
        base_directory + '/' + directory + '/', directory,
    )) for directory in os.listdir(base_directory)]
    pool.close()
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay


def read_csi_csv(base_directory, one_file=False):
    datax = None
    datay = None
    if one_file is True:
        files = os.listdir(base_directory)

        for f in files:
            df = pd.read_csv(os.path.join(base_directory, f))
            total_len = len(df)
            div_num = total_len // 64
            label = f.split('_')[0]
            lables = [label for _ in range(div_num)]

            if datay is None:
                datay = lables
            else:
                datay += lables

            for i in range(div_num):
                div = df.iloc[i*64:(i+1) * 64, :]
                if datax is None:
                    datax = np.array([div])
                else:
                    datax = np.vstack([datax, [np.array(div)]])

    else:
        #Todo
        print('Not implemented yet.')
        exit()
        datax = None
        datay = None

    return datax, np.array(datay)


def read_csi_from_pcap(pcap_dir):
    """
    Read pcap files and convert to dataframes.
    After that, concatenate all of them
    """
    datax = None
    datay = None

    pcap_files = os.listdir(pcap_dir)
    for pfile in pcap_files:
        filename = os.path.join(pcap_dir, pfile)
        df = pcap_to_df(filename, bandwidth=20)
        df = df.iloc[:64, :64]

        label = pfile.split('_')[0]

        if datax is None:
            datax = np.array([df])
            datay = np.array([label])
        else:
            datax = np.vstack([datax, [np.array(df)]])
            datay = np.concatenate([datay, np.array([label])])

    return datax, datay


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
        datax (np.array): dataset of dataloader dataframes
        datay (np.array): dataset of labels
    Returns:
        (dict) of:
          (torch.Tensor): sample of dataloader dataframes. Size (n_way, n_support+n_query, (dim))
          (int): n_way
          (int): n_support
          (int): n_query
    """
    #K = np.array(['empty', 'jump', 'stand', 'walk'])
    K = np.array(param['test_labels'])

    # extract support set & query set
    support_sample = []
    query_sample = []
    for cls in K:
        datax_cls = datax[datay == cls]
        support_cls = datax_cls[:n_support]
        query_cls = np.array(datax_cls[n_support:n_support+n_query])
        support_sample.append(support_cls)
        query_sample.append(query_cls)

    support_sample = np.array(support_sample)
    query_sample = np.array(query_sample)
    support_sample = torch.from_numpy(support_sample).float()
    query_sample = torch.from_numpy(query_sample).float()

    return ({
        's_csi_mats': support_sample,
        'q_csi_mats': query_sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


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
