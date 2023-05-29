import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from util import extract_train_sample, extract_test_sample


def train(model, learning_rate, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size, device):
    """
    Trains the protonet
    Args:
        model
        learning_rate
        train_x (np.array): dataloader dataframes of training set
        train_y(np.array): labels of training set
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        max_epoch (int): max epochs to train on
        epoch_size (int): episodes per epoch
    """
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_train_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.proto_train(sample)
            running_loss += output['loss']
            running_acc += output['acc']
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print('Epoch {:d} -- Acc: {:.5f} Loss: {:.9f}'.format(epoch + 1, epoch_acc, epoch_loss))

        # wandb.log({"acc": epoch_acc, "loss": epoch_loss})

        epoch += 1
        scheduler.step()

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode, device):
    """
    Tests the protonet
    Args:
        model: trained models
        test_x (np.array): dataloader dataframes of testing set
        test_y (np.array): labels of testing set
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        test_episode (int): number of episodes to test on
    """
    model = model.to(device)
    conf_mat = torch.zeros(n_way, n_way)
    running_loss = 0.0
    running_acc = 0.0

    '''
    Modified
    # Extract sample just once
    '''
    sample = extract_test_sample(n_way, n_support, n_query, test_x, test_y)
    query_samples = sample['q_csi_mats']

    # Create target domain Prototype Network with support set(target domain)
    z_proto = model.create_protoNet(sample)

    total_count = 0
    model.eval()
    with torch.no_grad():
        for episode in tqdm(range(test_episode), desc="test"):
            for label, q_samples in enumerate(query_samples):
                for i in range(0, len(q_samples) // n_way):
                    output = model.proto_test(q_samples[i * n_way:(i + 1) * n_way], z_proto, n_way, label)
                    # print(output)
                    a = output['y_hat'].cpu().int()
                    for cls in range(n_way):
                        conf_mat[cls, :] = conf_mat[cls, :] + torch.bincount(a[cls, :], minlength=n_way)
                    running_acc += output['acc']
                    total_count += 1
        print(conf_mat)
    avg_acc = running_acc / total_count
    print('Test results -- Acc: {:.5f}'.format(avg_acc))
    return (conf_mat / (test_episode * n_query), avg_acc)
