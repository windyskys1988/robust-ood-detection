from __future__ import print_function
import argparse
import os

import sys

sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import models.densenet as dn
import utils.svhn_loader as svhn
import numpy as np
import time
# import lmdb
from scipy import misc
from utils import ConfidenceLinfPGDAttack, MahalanobisLinfPGDAttack, softmax, metric, sample_estimator, \
    get_Mahalanobis_score, TinyImages

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str,
                    help='neural network name and training set')
parser.add_argument('--out-dataset', default="LSUN_resize", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--magnitude', default=0.0014, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')

parser.add_argument('--gpu', default='0', type=str,
                    help='gpu index')
parser.add_argument('--adv', help='adv ood evaluation', action='store_true')
parser.add_argument('--method', default='msp_and_odin', type=str, help='ood detection method')

parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon')
parser.add_argument('--iters', default=10, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=40, type=int,
                    help='mini-batch size')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--lmd', default=0.9, type=float, help='parameter lambda: weight of loss and KL')
parser.add_argument('--force', default=False, type=bool, help='force ODIN')
parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def MSP(outputs, model):
    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print(nnOutputs.shape)
    return nnOutputs


def tensor_stat(tag, arr):
    print(tag + " count ", arr.shape[0], " max ", torch.max(arr).data, " min ", torch.min(arr).data, " mean ",
          torch.mean(arr).data,
          " var ",
          torch.var(arr).data, " median ", torch.median(arr).data)


def DIS(inputs, outputs, model, model2):
    # 仅做距离评价，如需ODIN，需要用原eval_ood_detection
    output2 = model2(inputs)

    otkl = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(output2, dim=1), reduce=False)
    tokl = F.kl_div(F.log_softmax(output2, dim=1), F.softmax(outputs, dim=1), reduce=False)
    # print(otkl.size())
    # print(torch.mean(otkl+tokl).data)

    # output2 = output2.cpu().detach().numpy()
    # outputs = outputs.cpu().detach().numpy()
    # origin_max_scores = np.max(outputs, axis=1)
    # origin_max_indices = np.argmax(outputs, axis=1)
    # # 对应的得分
    # cor_scores = output2[tuple(np.arange(output2.shape[0])), tuple(origin_max_indices)]
    # diff_score = -np.abs(cor_scores - origin_max_scores)
    # # diff = output2 - outputs
    # print(np.mean(diff_score))

    return torch.div(1.0, (otkl + tokl + 0.00001)).cpu().detach().numpy()


def ODINDIS(inputs, outputs, model, model2, temper=1000, noiseMagnitude1=0.0014,forceODIN = args.force):
    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    criterion = nn.CrossEntropyLoss()
    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())

    # nat_input = inputs.detach().clone()
    # trans_input = torch.flip(nat_input, [-1])
    trans_input = Variable(inputs, requires_grad=True)

    output2 = model2(trans_input)

    outputs = outputs / temper
    output2 = output2 / temper

    nat_loss = criterion(outputs, labels)
    trans_loss = criterion(output2, labels)

    otkl = F.kl_div(F.softmax(outputs, dim=1), F.softmax(output2, dim=1), reduce=False)
    tokl = F.kl_div(F.softmax(output2, dim=1), F.softmax(outputs, dim=1), reduce=False)
    # print(otkl.size())
    # print(torch.mean(torch.sum(otkl, dim=1)))
    # print(torch.mean(torch.sum(tokl, dim=1)))
    # exit(0)
    lmd = args.lmd
    loss = lmd * (nat_loss + trans_loss) + (1 - lmd) * (
            torch.mean(torch.sum(otkl, dim=1)) + torch.mean(torch.sum(tokl, dim=1)))

    loss.backward()
    # 普通 ODIN
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # tesnsor_stat('grad', gradient)

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    if forceODIN:
        return nnOutputs

    # ODIN 2
    gradient2 = torch.ge(trans_input.grad.data, 0)
    gradient2 = (gradient2.float() - 0.5) * 2
    # tensor_stat('grad', gradient - gradient2)

    # Adding small perturbations to images
    tempInput2 = torch.add(trans_input.data, -noiseMagnitude1, gradient2)
    output2 = model(Variable(tempInput2))
    output2 = output2 / temper
    # Calculating the confidence after adding perturbations
    nnOutput2 = output2.data.cpu().numpy()
    nnOutput2 = nnOutput2 - np.max(nnOutput2, axis=1, keepdims=True)
    nnOutput2 = np.exp(nnOutput2) / np.sum(np.exp(nnOutput2), axis=1, keepdims=True)
    # return nnOutput2

    # print(nnOutputs)
    # print(nnOutput2)
    # arr_stat('nn1', nnOutputs)
    # arr_stat('nn2', nnOutput2)

    nnOutputs = torch.Tensor(nnOutputs)
    nnOutput2 = torch.Tensor(nnOutput2)
    # print(F.softmax(nnOutputs, dim=1))
    # print(nnOutput2.size())
    otkl = F.kl_div(nnOutputs, nnOutput2, reduce=False)
    tokl = F.kl_div(nnOutput2, nnOutputs, reduce=False)
    # print(otkl.size())
    # tensor_stat('df', otkl)

    return torch.div(1.0, (otkl + tokl + 0.00001)).cpu().detach().numpy()


def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + args.in_dataset)
    print('out_distribution: ' + args.out_dataset)
    print('Model Name: ' + args.name)
    print('Under attack: ' + str(args.adv))
    print('')

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100. * results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100. * results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100. * results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100. * results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100. * results[stype]['AUOUT']), end='')
        print('')


def eval_msp_and_odin():
    stypes = ['MSP', 'ODIN']

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
    model2 = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

    checkpoint = torch.load(
        "./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint['state_dict'])
    model2.load_state_dict(checkpoint['state_dict2'])

    model.eval()
    model2.eval()
    model.cuda()
    model2.cuda()

    if args.out_dataset == 'SVHN':
        testsetout = svhn.SVHN('../../data/SVHN/', split='test',
                               transform=transforms.ToTensor(), download=False)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)
    elif args.out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root="../../data/dtd/images",
                                                      transform=transforms.Compose(
                                                          [transforms.Resize(32), transforms.CenterCrop(32),
                                                           transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=2)
    elif args.out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
                                                      transform=transforms.Compose(
                                                          [transforms.Resize(32), transforms.CenterCrop(32),
                                                           transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=2)
    else:
        testsetout = torchvision.datasets.ImageFolder("../../data/{}".format(args.out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)

    t0 = time.time()
    f1 = open(os.path.join(save_dir, "confidence_MSP_In.txt"), 'w')
    f2 = open(os.path.join(save_dir, "confidence_MSP_Out.txt"), 'w')
    g1 = open(os.path.join(save_dir, "confidence_ODIN_In.txt"), 'w')
    g2 = open(os.path.join(save_dir, "confidence_ODIN_Out.txt"), 'w')
    N = 10000
    if args.out_dataset == "iSUN": N = 8925
    if args.out_dataset == "dtd": N = 5640
    ########################################In-distribution###########################################
    print("Processing in-distribution images")
    if args.adv:
        attack = ConfidenceLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
                                         eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
                                         in_distribution=True, num_classes=num_classes)

    count = 0
    for j, data in enumerate(testloaderIn):
        if j == 10: break
        images, _ = data
        batch_size = images.shape[0]

        if count + batch_size > N:
            images = images[:N - count]
            batch_size = images.shape[0]

        if args.adv:
            adv_images = attack.perturb(images)
            inputs = Variable(adv_images, requires_grad=True)
        else:
            inputs = Variable(images, requires_grad=True)

        outputs = model(inputs)

        nnOutputs = MSP(outputs, model)

        for k in range(batch_size):
            f1.write("{}\n".format(np.max(nnOutputs[k])))

        # torch.save(outputs, args.name + '_in.pth')
        nnOutputs = ODINDIS(inputs, outputs, model, model2)

        for k in range(batch_size):
            g1.write("{}\n".format(np.max(nnOutputs[k])))
            # g1.write("{}\n".format(np.mean(nnOutputs[k])))
            # max or mean

        count += batch_size
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
        t0 = time.time()

        if count == N: break

    ###################################Out-of-Distributions#####################################
    t0 = time.time()
    print("Processing out-of-distribution images")
    if args.adv:
        attack = ConfidenceLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
                                         eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
                                         in_distribution=False, num_classes=num_classes)
    count = 0

    for j, data in enumerate(testloaderOut):
        if j == 10: break
        images, labels = data
        batch_size = images.shape[0]

        if args.adv:
            adv_images = attack.perturb(images)
            inputs = Variable(adv_images, requires_grad=True)
        else:
            inputs = Variable(images, requires_grad=True)

        # print(inputs.size())
        outputs = model(inputs)
        # print(outputs.size())
        nnOutputs = MSP(outputs, model)

        for k in range(batch_size):
            f2.write("{}\n".format(np.max(nnOutputs[k])))

        # torch.save(outputs, args.name + '_' + args.out_dataset + '_out.pth')
        nnOutputs = ODINDIS(inputs, outputs, model, model2)
        # exit(0)

        for k in range(batch_size):
            g2.write("{}\n".format(np.max(nnOutputs[k])))
            # g2.write("{}\n".format(np.mean(nnOutputs[k])))

        count += batch_size
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
        t0 = time.time()

        if count == N: break

    f1.close()
    f2.close()
    g1.close()
    g2.close()

    results = metric(save_dir, stypes)

    print_results(results, stypes)


if __name__ == '__main__':
    if args.method == 'msp_and_odin':
        eval_msp_and_odin()
