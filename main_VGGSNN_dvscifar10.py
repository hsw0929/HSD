import torch
import torch.multiprocessing as mp
import argparse
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d,replace_batchnorm_with_tdbatchnorm
from ImageNet.train import main_worker
import torch.nn as nn
import os
from torchvision import datasets, transforms
from modules import TCL, MyFloor, ScaledNeuron, StraightThrough,LIFSpike
from Models.VGG_CIFAR10_DVS import *
from torch.utils.data import DataLoader
from Preprocess.preprocess_dvscifar10 import DVSCIFAR10
torch.backends.cudnn.enabled = False
import torch.utils.data as data
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cuda:0"

def build_dvscifar10(transform=False):

    train_dataset = DVSCIFAR10(transform=True)
    val_dataset = DVSCIFAR10(data_type='test', transform=False)

    return train_dataset, val_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=64, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=64, type=int,
                        help='Training epochs')  # better if set to 300 for CIFAR dataset
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    #parser.add_argument('--device', default='cuda:5', type=str, help='cuda or cpu')
    parser.add_argument('--l', default=4, type=int, help='L')
    parser.add_argument('--t', default=4, type=int, help='T')
    #parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--data', type=str, default='cifar10dvs')
    #parser.add_argument('--model', type=str, default='vgg11')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    args = parser.parse_args()

    seed_all()
    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        # preparing data

        train_dataset, val_dataset=build_dvscifar10(transform=False)

        workers = 16
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                                   num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                                  num_workers=workers, pin_memory=True)
        #preparing model

        model=VGGNet(num_classes=10)
        model = replace_activation_by_floor(model, t=args.l)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = None

        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        else:
            raise NotImplementedError(args.opt)

        if args.action == 'train':
            train_ann_cifar10_dvs(train_loader, test_loader,optimizer, model, args.epochs, device, criterion, args.lr, args.wd, args.id)

        elif args.action == 'test' or args.action == 'evaluate':

            large_model=model
            large_model.load_state_dict(torch.load('./saved_models_std_ann_snn_l_16_dvs/' + args.id + '.pth',map_location=device))
            print('large_model:',large_model)

            model.load_state_dict(torch.load('./saved_models_std_ann_snn_l_16_dvs/' + args.id + '.pth',map_location=device))
            small_model=replace_activation_by_neuron1(model)

            print('small_model:',small_model)

            max_acc=0.0
            max_activate=0.0

            large_model.to(device)
            small_model.to(device)

            train_loss, train_acc, test_loss, test_acc = train_snn_step(large_model,small_model, args.device, train_loader, test_loader, criterion,args.epochs, lr=1e-5, wd=5e-4)




