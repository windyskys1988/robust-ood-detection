from __future__ import print_function
import argparse
import os

import sys

sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import models.densenet as dn
import numpy as np
import time

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--out-dataset', default="LSUN_resize", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--name', required=True, type=str,
                    help='neural network name and training set')
parser.add_argument('--adv', help='adv ood evaluation', action='store_true')

parser.add_argument('--gpu', default='0', type=str,
                    help='gpu index')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=40, type=int,
                    help='mini-batch size')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 50)')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)



def tesnsor_stat(tag, arr):
    print(tag + " count ", arr.shape[0], " max ", torch.max(arr), " min ", torch.min(arr), " mean ", torch.mean(arr),
          " var ",
          torch.var(arr), " median ", torch.median(arr))


def eval_acc():
    print('test accuracy')

    save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.name, 'adv' if args.adv else 'nat')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = time.time()
    # loading data sets
    normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=2)
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True,
                                                transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=2)
        num_classes = 100

    model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

    checkpoint = torch.load(
        "./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()

    # 纯噪声数据测试
    # num_gaussian_inputs = 10
    # gaussian_inputs = torch.rand((num_gaussian_inputs, 3, 32, 32))
    # print(gaussian_inputs.size())
    # output = model(gaussian_inputs)
    criterion = nn.CrossEntropyLoss()

    for i, (input, target) in enumerate(testloaderIn):
        target = target.cuda()

        add_num = int(target.size()[0] / 7)

        nat_input = input.detach().clone()

        nat_output = model(nat_input)
        nat_loss = criterion(nat_output, target)

        # measure accuracy and record loss
        nat_prec1 = accuracy(nat_output.data, target, topk=(1,))[0]
        nat_losses.update(nat_loss.data, input.size(0))
        nat_top1.update(nat_prec1, input.size(0))

        # # compute gradient and do SGD step
        # loss = nat_loss
        # if args.lr_scheduler == 'cosine_annealing':
        #     scheduler.step()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        #
        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        if i % args.print_freq == 0 or i == len(testloaderIn) - 1:
            print('Epoch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(testloaderIn),
                loss=nat_losses, top1=nat_top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    eval_acc()
