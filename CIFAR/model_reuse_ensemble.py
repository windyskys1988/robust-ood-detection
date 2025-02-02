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
import torch.nn.functional as F

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
        num_classes = 5
    elif args.in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True,
                                                transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=2)
        num_classes = 100

    model1 = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
    model2 = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)

    checkpoint1 = torch.load(
        "./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name + '04', epochs=args.epochs))
    model1.load_state_dict(checkpoint1['state_dict'])

    model1.eval()
    model1.cuda()

    checkpoint2 = torch.load(
        "./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name + '59', epochs=args.epochs))
    model2.load_state_dict(checkpoint2['state_dict'])

    model2.eval()
    model2.cuda()

    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()

    # 纯噪声数据测试
    # num_gaussian_inputs = 10
    # gaussian_inputs = torch.rand((num_gaussian_inputs, 3, 32, 32))
    # print(gaussian_inputs.size())
    # output = model(gaussian_inputs)
    criterion = nn.CrossEntropyLoss()

    tfparams = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.1]],
        [[1.0, 0.0, 0.1], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, -0.1]],
        [[1.0, 0.0, -0.1], [0.0, 1.0, 0.0]],
        [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0]],
        [[0.9, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.9, 0.0]],
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.1, 1.0, 0.0]],
        [[1.0, -0.1, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [-0.1, 1.0, 0.0]],
    ])

    affine_params = torch.from_numpy(tfparams).float()

    for i, (input, target) in enumerate(testloaderIn):
        target = target.cuda()
        # print(target)

        nat_input = input.detach().clone()
        input1 = Variable(nat_input, requires_grad=True)
        input2 = Variable(nat_input, requires_grad=True)

        output1 = model1(input1)
        output2 = model2(input2)

        affine_outputs1 = []
        affine_outputs2 = []
        for j, affine_param in enumerate(affine_params):
            # print(affine_param)
            p2 = nat_input.size()
            p1 = affine_param.repeat(p2[0], 1, 1)

            grid = F.affine_grid(p1, p2)
            trans_data = F.grid_sample(nat_input, grid, padding_mode='reflection')
            trans_input = Variable(trans_data.data.cpu().cuda(0), requires_grad=True)

            trans_output1 = model1(trans_input)
            trans_output2 = model2(trans_input)
            # affine_outputs1.append(trans_output1)
            affine_outputs1.append(F.softmax(trans_output1))
            # affine_outputs2.append(trans_output2)
            affine_outputs2.append(F.softmax(trans_output2))

        affine_outputs1 = torch.stack(affine_outputs1)
        affine_outputs2 = torch.stack(affine_outputs2)
        # affine_outputs_sum1 = torch.sum(affine_outputs1, dim=0)
        affine_outputs_sum1 = -torch.var(affine_outputs1, dim=0)
        # affine_outputs_sum2 = torch.sum(affine_outputs2, dim=0)
        affine_outputs_sum2 = -torch.var(affine_outputs2, dim=0)
        score1 = torch.max(affine_outputs_sum1, dim=1).values
        score2 = torch.max(affine_outputs_sum2, dim=1).values
        # print(score1)
        # print(score2)

        mask = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0] if x else [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] for x in score1 > score2]
        # for x in odin1 > odin2:
        #     if x:
        #         mask.append([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        #     else:
        #         mask.append([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # print(mask)

        # exit(0)

        torch.set_printoptions(precision=8, sci_mode=False)
        # print(output1.size())
        # print(output2.size())

        nat_output = torch.cat((output1, output2), dim=1) * torch.tensor(mask).float().cuda()
        # print(nat_output)
        nat_output = F.softmax(nat_output, dim=1)
        # print(nat_output)
        # exit(0)
        # print(nat_output.size())
        # print(nat_output[:10])
        # print(torch.argmax(torch.cat((output1, output2), dim=1), dim=1))
        # print(torch.argmax(nat_output, dim=1))
        # print(target[:10])
        # exit(0)

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

# from __future__ import print_function
# import argparse
# import os
#
# import sys
#
# sys.path.append("..")
#
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# # import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegressionCV
# import models.densenet as dn
# import utils.svhn_loader as svhn
# import numpy as np
# import time
# # from utils import ConfidenceLinfPGDAttack, MahalanobisLinfPGDAttack, softmax, metric, sample_estimator, \
# #     get_Mahalanobis_score, TinyImages
#
# parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
#
# parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
# parser.add_argument('--name', required=True, type=str,
#                     help='neural network name and training set')
# parser.add_argument('--out-dataset', default="LSUN_resize", type=str,
#                     help='out-of-distribution dataset')
# parser.add_argument('--magnitude', default=0.0014, type=float,
#                     help='perturbation magnitude')
# parser.add_argument('--temperature', default=1000, type=int,
#                     help='temperature scaling')
#
# parser.add_argument('--gpu', default='0', type=str,
#                     help='gpu index')
# parser.add_argument('--method', default='msp_and_odin', type=str, help='ood detection method')
#
# parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon')
# parser.add_argument('--iters', default=10, type=int,
#                     help='attack iterations')
# parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')
# parser.add_argument('--epochs', default=100, type=int,
#                     help='number of total epochs to run')
#
# parser.add_argument('-b', '--batch-size', default=40, type=int,
#                     help='mini-batch size')
#
# parser.add_argument('--layers', default=100, type=int,
#                     help='total number of layers (default: 100)')
#
# parser.set_defaults(argument=True)
#
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)
#
#
# def MSP(outputs, model):
#     # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
#     nnOutputs = outputs.data.cpu()
#     nnOutputs = nnOutputs.numpy()
#     nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
#     nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
#     return nnOutputs
#
#
# def tesnsor_stat(tag, arr):
#     print(tag + " count ", arr.shape[0], " max ", torch.max(arr), " min ", torch.min(arr), " mean ", torch.mean(arr),
#           " var ",
#           torch.var(arr), " median ", torch.median(arr))
#
#
# def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
#     # Calculating the perturbation we need to add, that is,
#     # the sign of gradient of cross entropy loss w.r.t. input
#     # print(torch.softmax(outputs,axis=1))
#     # return 0
#
#     criterion = nn.CrossEntropyLoss()
#
#     maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
#
#     # Using temperature scaling
#     outputs = outputs / temper
#     labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
#     loss = criterion(outputs, labels)
#     loss.backward()
#
#     # Normalizing the gradient to binary in {0, 1}
#     # gradient = inputs.grad.data * (40000.0) # 5000 best
#     # 绝对值大小对比
#     # th_top = torch.median(inputs.grad.data[inputs.grad.data > 0])
#     # # print(th_top)
#     # th_bottom = torch.median(inputs.grad.data[inputs.grad.data < 0])
#     # # print(th_bottom)
#     # gradient_top = torch.ge(inputs.grad.data, th_top)
#     # gradient_bottom = torch.le(inputs.grad.data, th_bottom)
#     # gradient_great = gradient_top.float() - gradient_bottom.float()
#     # # 取较大绝对值
#     # gradient = gradient_great * 0.5
#     # print(gradient_great)
#
#     # 取较小绝对值
#     # gradient = torch.ge(inputs.grad.data, 0)
#     # gradient = (gradient.float() - 0.5) * 2
#     # gradient = (gradient - gradient_great) * 6.0
#     # print(gradient)
#
#     # print(gradient.float())
#     # 普通 ODIN
#     gradient = torch.ge(inputs.grad.data, 0)
#     gradient = (gradient.float() - 0.5) * 2
#     # tesnsor_stat('grad', gradient)
#
#     # Adding small perturbations to images
#     tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
#     outputs = model(Variable(tempInputs))
#     outputs = outputs / temper
#     # Calculating the confidence after adding perturbations
#     nnOutputs = outputs.data.cpu()
#     nnOutputs = nnOutputs.numpy()
#     nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
#     nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
#
#     return nnOutputs
#
#
# def print_results(results, stypes):
#     mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
#
#     print('in_distribution: ' + args.in_dataset)
#     print('out_distribution: ' + args.out_dataset)
#     print('Model Name: ' + args.name)
#     print('Under attack: ' + str(args.adv))
#     print('')
#
#     for stype in stypes:
#         print(' OOD detection method: ' + stype)
#         for mtype in mtypes:
#             print(' {mtype:6s}'.format(mtype=mtype), end='')
#         print('\n{val:6.2f}'.format(val=100. * results[stype]['FPR']), end='')
#         print(' {val:6.2f}'.format(val=100. * results[stype]['DTERR']), end='')
#         print(' {val:6.2f}'.format(val=100. * results[stype]['AUROC']), end='')
#         print(' {val:6.2f}'.format(val=100. * results[stype]['AUIN']), end='')
#         print(' {val:6.2f}\n'.format(val=100. * results[stype]['AUOUT']), end='')
#         print('')
#
#
# def eval_msp_and_odin():
#     stypes = ['MSP', 'ODIN']
#
#     save_dir = os.path.join('output/ood_scores/', args.out_dataset, args.name, 'adv' if args.adv else 'nat')
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     start = time.time()
#     # loading data sets
#     normalizer = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#
#     if args.in_dataset == "CIFAR-10":
#         testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
#         testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
#                                                    shuffle=True, num_workers=2)
#         num_classes = 5
#     elif args.in_dataset == "CIFAR-100":
#         testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True,
#                                                 transform=transform)
#         testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
#                                                    shuffle=True, num_workers=2)
#         num_classes = 100
#
#     model1 = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
#     model2 = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
#
#     checkpoint1 = torch.load(
#         "./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name+'04', epochs=args.epochs))
#     model1.load_state_dict(checkpoint1['state_dict'])
#
#     model1.eval()
#     model1.cuda()
#
#     checkpoint2 = torch.load(
#         "./checkpoints/{name}/checkpoint_{epochs}.pth.tar".format(name=args.name+'59', epochs=args.epochs))
#     model2.load_state_dict(checkpoint2['state_dict'])
#
#     model2.eval()
#     model2.cuda()
#
#     if args.out_dataset == 'SVHN':
#         testsetout = svhn.SVHN('../../data/SVHN/', split='test',
#                                transform=transforms.ToTensor(), download=False)
#         testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
#                                                     shuffle=True, num_workers=2)
#     elif args.out_dataset == 'dtd':
#         testsetout = torchvision.datasets.ImageFolder(root="../../data/dtd/images",
#                                                       transform=transforms.Compose(
#                                                           [transforms.Resize(32), transforms.CenterCrop(32),
#                                                            transforms.ToTensor()]))
#         testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
#                                                     num_workers=2)
#     elif args.out_dataset == 'places365':
#         testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
#                                                       transform=transforms.Compose(
#                                                           [transforms.Resize(32), transforms.CenterCrop(32),
#                                                            transforms.ToTensor()]))
#         testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True,
#                                                     num_workers=2)
#     else:
#         testsetout = torchvision.datasets.ImageFolder("../../data/{}".format(args.out_dataset), transform=transform)
#         testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
#                                                     shuffle=True, num_workers=2)
#
#     t0 = time.time()
#     f1 = open(os.path.join(save_dir, "confidence_MSP_In.txt"), 'w')
#     f2 = open(os.path.join(save_dir, "confidence_MSP_Out.txt"), 'w')
#     g1 = open(os.path.join(save_dir, "confidence_ODIN_In.txt"), 'w')
#     g2 = open(os.path.join(save_dir, "confidence_ODIN_Out.txt"), 'w')
#     N = 10000
#     if args.out_dataset == "iSUN": N = 8925
#     if args.out_dataset == "dtd": N = 5640
#     ########################################In-distribution###########################################
#     print("Processing in-distribution images")
#
#     count = 0
#     for j, data in enumerate(testloaderIn):
#         if j == 10: break
#         images, _ = data
#         print(_)
#         batch_size = images.shape[0]
#
#         if count + batch_size > N:
#             images = images[:N - count]
#             batch_size = images.shape[0]
#
#
#         inputs = Variable(images, requires_grad=True)
#
#         outputs1 = model1(inputs)
#         outputs2 = model2(inputs)
#
#         print(outputs1.size())
#         print(outputs2.size())
#
#         # nnOutputs = MSP(outputs, model)
#         #
#         # for k in range(batch_size):
#         #     f1.write("{}\n".format(np.max(nnOutputs[k])))
#         #
#         # nnOutputs = ODIN(inputs, outputs, model, temper=args.temperature, noiseMagnitude1=args.magnitude)
#         #
#         # for k in range(batch_size):
#         #     g1.write("{}\n".format(np.max(nnOutputs[k])))
#         #
#         # count += batch_size
#         # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
#         # t0 = time.time()
#         #
#         # if count == N: break
#
#     ###################################Out-of-Distributions#####################################
#     # t0 = time.time()
#     # print("Processing out-of-distribution images")
#     # if args.adv:
#     #     attack = ConfidenceLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
#     #                                      eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1.,
#     #                                      in_distribution=False, num_classes=num_classes)
#     # count = 0
#     #
#     # for j, data in enumerate(testloaderOut):
#     #     if j == 10: break
#     #     images, labels = data
#     #     print(labels)
#     #     batch_size = images.shape[0]
#     #
#     #     if args.adv:
#     #         adv_images = attack.perturb(images)
#     #         inputs = Variable(adv_images, requires_grad=True)
#     #     else:
#     #         inputs = Variable(images, requires_grad=True)
#     #
#     #     outputs = model(inputs)
#     #
#     #     nnOutputs = MSP(outputs, model)
#     #
#     #     for k in range(batch_size):
#     #         f2.write("{}\n".format(np.max(nnOutputs[k])))
#     #
#     #     nnOutputs = ODIN(inputs, outputs, model, temper=args.temperature, noiseMagnitude1=args.magnitude)
#     #
#     #     for k in range(batch_size):
#     #         g2.write("{}\n".format(np.max(nnOutputs[k])))
#     #
#     #     count += batch_size
#     #     print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
#     #     t0 = time.time()
#     #
#     #     if count == N: break
#     #
#     # f1.close()
#     # f2.close()
#     # g1.close()
#     # g2.close()
#     #
#     # results = metric(save_dir, stypes)
#     #
#     # print_results(results, stypes)
#
#
#
#
# if __name__ == '__main__':
#     eval_msp_and_odin()
