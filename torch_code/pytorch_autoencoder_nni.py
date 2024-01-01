from __future__ import print_function
import argparse
import time
import os
import sys
import datetime
import math
import numpy as np
import logging
import random
from tds import TDS
is_nni = False
if is_nni:
    import nni

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

strhdlr = logging.StreamHandler()
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision as tv

from utils import *
import kfac


def initialize():
    # Training Parameters
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='autoencoder')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1k)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1k)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')

    # SGD/Adam/tds Parameters
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-schedule', type=str, default='linear', 
                        choices=['step', 'linear', 'polynomial', 'cosine'], help='learning rate schedules')
    parser.add_argument('--momentum', type=float, default=0.43, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='W',
                        help='SGD weight decay (default: 0.0)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--opt-name', type=str, default='sgd', metavar='WE',
                        help='choose base optimizer [sgd, adam, tds]')

    # KFAC Parameters
    parser.add_argument('--kfac-type', type=str, default='Femp', 
                        help='choices: F1mc or Femp') 
    parser.add_argument('--kfac-name', type=str, default='kfac',
                        help='choices: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
                        help='choices: ComputeFactor,CommunicateFactor,ComputeInverse,CommunicateInverse')
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 for no kfac updates) (default: 20)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.6,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.005,
                        help='KFAC damping factor (defaultL 0.03)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')

    # Other Parameters
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for tds, default is 1e-8')
    parser.add_argument('--log-dir', default='./logs',
                        help='log directory')
    parser.add_argument('--dir', default='./datasets',
                        help='location of the training dataset in the local filesystem (will be downloaded if needed)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    args = parser.parse_args()


    # Training Settings
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ############ NNI Start ################
    if is_nni:
        optimized_params = nni.get_next_parameter()
        args.opt_name = optimized_params['opt_name']
        args.momentum = 1.0 - optimized_params['momentum']
        args.stat_decay = 1.0 - optimized_params['stat_decay']
        args.base_lr = optimized_params['base_lr']
        if optimized_params['opt_name'] == 'sgd':
            args.kfac_name = optimized_params['kfac_name']
            args.damping = optimized_params['damping']
            args.kl_clip = optimized_params['kl_clip']
        if optimized_params['opt_name'] == 'tds':
            args.eps = optimized_params['eps']
    ############ NNI End ################

    # Logging Settings
    args.use_kfac = True if (args.kfac_update_freq > 0 and args.opt_name == 'sgd') else False
    algo = args.kfac_name if args.use_kfac else args.opt_name
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(args.log_dir,
        '{}_{}_ep{}_bs{}_lr{}_gpu{}_kfac{}_{}_{}_clip{}.log'.format(args.dataset, args.model, args.epochs, args.batch_size, args.base_lr, 1, args.kfac_update_freq, algo, args.lr_schedule, args.kl_clip))

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    args.verbose = True
    
    if args.verbose:
        logger.info("torch version: %s", torch.__version__)
        logger.info(args)
    return args


def get_dataset(args):
    # Load MNIST
    # kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    kwargs = {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(dtype=torch.float32)])
    if args.dataset == 'mnist':
        train_dataset = tv.datasets.MNIST(
            root='../data/', download=True,  train=True, transform=transform)
        test_dataset = tv.datasets.MNIST(
            root='../data/', download=True,  train=True, transform=transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=5*args.batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader


class Autoencoder(nn.Module):
    def __init__(self, dataset):
        super(Autoencoder, self).__init__()
        in_dim = 625 if dataset == 'faces' else 784
        self.fc1 = nn.Linear(in_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 30)
        self.fc5 = nn.Linear(30, 250)
        self.fc6 = nn.Linear(250, 500)
        self.fc7 = nn.Linear(500, 1000)
        self.fc8 = nn.Linear(1000, in_dim)
        self.r = F.relu
   
    def forward(self, inputs):
        # encoder
        x = inputs.view(inputs.size(0), -1)
        x = self.r(self.fc1(x))
        x = self.r(self.fc2(x))
        x = self.r(self.fc3(x))
        x = self.fc4(x)
        # decoder
        x = self.r(self.fc5(x))
        x = self.r(self.fc6(x))
        x = self.r(self.fc7(x))
        x = self.fc8(x)
        return x

def get_model(args):
    model = Autoencoder(args.dataset)

    if args.cuda:
        model.cuda()

    # Optimizer
    criterion = nn.BCEWithLogitsLoss(
        weight=None, size_average=None, reduce=False, reduction='none', pos_weight=None)

    if args.opt_name == "adam":
        print("Giving adam optimizer")
        optimizer = optim.Adam(model.parameters(), 
                lr=args.base_lr, 
                betas=(0.9, 0.999))
    elif args.opt_name == "tds":
        print("Giving tds optimizer")
        optimizer = TDS(model.parameters(), 
                    lr=args.base_lr, 
                    betas=(args.momentum, args.stat_decay),
                    eps=args.eps,
                    weight_decay=args.weight_decay)
    elif args.opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                lr=args.base_lr, 
                momentum=args.momentum,
                weight_decay=args.weight_decay)

    if args.use_kfac and args.opt_name=='sgd':
        KFAC = kfac.get_kfac_module(args.kfac_name)
        preconditioner = KFAC(model, 
                lr=args.base_lr, 
                factor_decay=args.stat_decay, 
                damping=args.damping, 
                kl_clip=args.kl_clip, 
                fac_update_freq=args.kfac_cov_update_freq, 
                kfac_update_freq=args.kfac_update_freq,
                exclude_parts=args.exclude_parts)
    else:
        preconditioner = None

    # Learning Rate Schedule
    lr_vec = np.concatenate([np.linspace(0, args.base_lr, args.warmup_epochs),
        np.linspace(args.base_lr, 0, args.epochs-args.warmup_epochs+2)[1:-1]], axis=0)

    return model, optimizer, preconditioner, lr_vec, criterion

def train(epoch, model, optimizer, preconditioner, lr_vec, criterion, train_loader, args):
    model.train()
    for g in optimizer.param_groups:
        g['lr'] = lr_vec[epoch]
    if args.use_kfac:
        for g in preconditioner.param_groups:
            g['lr'] = lr_vec[epoch]

    train_loss = Metric('train_loss')
    display = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        stime = time.time()

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))

        loss = criterion(output, data.view(data.size(0), -1)).sum(dim=1).mean(dim=0)
        loss.backward()
        
        with torch.no_grad():
            train_loss.update(loss)

        if args.use_kfac:
            preconditioner.step(epoch=epoch)
    
        optimizer.step()
        print("Epoch/batch [%d][%d] train loss: %.6f" % (epoch, batch_idx, loss))

    if args.verbose:
        logger.info("[%d] epoch train loss: %.4f" % (epoch, train_loss.avg.item()))

    if args.verbose:
        logger.info("[%d] epoch learning rate: %f" % (epoch, optimizer.param_groups[0]['lr']))

def test(epoch, model, criterion, test_loader, args):
    model.eval()
    test_loss = Metric('val_loss')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            
            test_loss.update(criterion(output, data.view(data.size(0), -1)).sum(1).mean(0))
    if is_nni:
        print("putting results in nni")
        nni.report_intermediate_result({ 'default':test_loss.avg.item()})
        if epoch == 99:
            nni.report_final_result({'default':test_loss.avg.item()})
    if args.verbose:
        logger.info("[%d] evaluation loss: %.4f" % (epoch, test_loss.avg.item()))


if __name__ == '__main__':
    args = initialize()

    train_loader, test_loader = get_dataset(args)
    args.num_steps_per_epoch = len(train_loader)
    model, optimizer, preconditioner, lr_vec, criterion = get_model(args)

    start = time.time()

    for epoch in range(args.epochs):
        stime = time.time()
        train(epoch, model, optimizer, preconditioner, lr_vec, criterion, train_loader, args)
        if args.verbose:
            logger.info("[%d] epoch train time: %.3f"%(epoch, time.time() - stime))
        if (epoch+1)%40 == 0 or epoch==99:
            stime = time.time()
            test(epoch, model, criterion, test_loader, args)
            if args.verbose:
                logger.info("[%d] epoch valid time: %.3f"%(epoch, time.time() - stime))

    if args.verbose:
        logger.info("Total Training + Valid Time: %s", str(datetime.timedelta(seconds=time.time() - start)))

