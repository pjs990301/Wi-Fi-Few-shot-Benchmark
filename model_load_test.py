from util import load_UT_HAR_supervised_model, load_ReWiS_supervised_model
from ReWiS_model import ReWiS_LeNet
import torch

if __name__ == "__main__" : 
    model = ReWiS_LeNet()
    model.load_state_dict(torch.load('model.pt'))
    print(model)