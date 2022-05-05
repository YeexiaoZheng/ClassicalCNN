import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from LeNet import LeNet
from AlexNet import AlexNet
from VGGNet import VGGNet
from GoogLeNet import GoogLeNet
from ResNet import ResNet

from Plot import myplt

import argparse

SPLIT_RATE = 0.8
PRINT = 50
BATCH_SIZE = 64
TRANSFORM = transforms.ToTensor()

def test(model, test_loader):
    model.eval()
    cor, all = 0, 0
    for (x, y) in test_loader:
        all += len(y)
        scores = model(x)
        for idx, each in enumerate(scores):
            if y[idx] == torch.argmax(each): 
                cor += 1

    acc = cor / all
    print('acc: ', acc)
    return acc

def train(model, loss_func, optimizer, train_loader, val_loader=None, epoch=15):
    accs = []
    losss = []
    for e in range(epoch):
        for idx, (x, y) in enumerate(train_loader):
            # switch to train mode
            model.train()
            scores = model(x)
            loss = loss_func(scores, y)

            optimizer.zero_grad()
            loss.backward()
            # clip_grad.clip_grad_norm_(model.parameters(), max_norm=20)
            optimizer.step()
            
            if idx % PRINT == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, idx, loss.item()))
                losss.append(float(loss))
                if val_loader:
                    accs.append(test(model, val_loader))

    return accs, losss

if __name__ == '__main__':

    mnist = MNIST('../datasets', train=True, transform=TRANSFORM, download=True)
    test_data = MNIST('../datasets', train=False, transform=TRANSFORM, download=True)
    train_data = mnist
    # train_data, val_data = random_split(mnist, [int(SPLIT_RATE * len(mnist)), len(mnist) - int(SPLIT_RATE * len(mnist))])
    # train_data, val_data = random_split(mnist, [len(mnist)-len(test_data), len(test_data)])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    accs_list = []
    losss_list = []
    test_acc_list = []

    epoch = 5
    plot = True

    model_list = [
        (LeNet(), 0.001, 0), 
        (AlexNet(), 0.001, 0.2), 
        (VGGNet(), 0.0005, 0.4), 
        (GoogLeNet(), 0.0005, 0.2), 
        (ResNet(), 0.0005, 0)
    ]

    for model, lr, dp in model_list:
        print(model, lr, dp)

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        accs, losss = train(model, loss_func, optimizer, train_loader, epoch=epoch)
        accs_list.append(accs)
        losss_list.append(losss)

        print()
        print('test acc: ')
        test_acc_list.append(test(model, test_loader))
        print()

    if plot:
        # myplt.line3dim(
        #     model_type='all kind model',
        #     xlabel='epoch', xlist=[str(i) for i in range(1, epoch + 1)],
        #     ylabel='val_acc', ylist=accs_list,
        #     zlabel='lr', zlist=[(str(model).split('(')[0], lr, dp) for model, lr, dp in model_list]
        # )

        myplt.line3dim(
            model_type='all kind model',
            xlabel='epoch', xlist=[str(i) for i in range(1, epoch + 1)],
            ylabel='loss', ylist=losss_list,
            zlabel='lr', zlist=[(str(model).split('(')[0], lr, dp) for model, lr, dp in model_list]
        )

        myplt.bar(
            model_type='all kind model',
            xlabel='learning_rate', xlist=[(str(model).split('(')[0], lr, dp) for model, lr, dp in model_list],
            ylabel='test_acc', ylist=test_acc_list,
        )