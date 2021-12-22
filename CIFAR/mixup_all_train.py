import argparse
import os

import sys

sys.path.append("..")

import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np

import models.densenet as dn
from utils import LinfPGDAttack, TinyImages

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--adv', action='store_true', help='adversarial training')
parser.add_argument('--adv-only-in', action='store_true', help='adversarial training only on in-distribution data')
parser.add_argument('--ood', action='store_true', help='training with ood samples')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')

parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon')
parser.add_argument('--iters', default=2, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')

parser.add_argument('--beta1', default=1.0, type=float, help='beta1 for adv_in_loss')
parser.add_argument('--beta2', default=0.5, type=float, help='beta2 for nat_out_loss')
parser.add_argument('--beta3', default=0.5, type=float, help='beta3 for adv_out_loss')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--save-epoch', default=10, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--ood-batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lr-scheduler', default='cosine_annealing', help='learning rate scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', required=True, type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/%s/" % (args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# torch.manual_seed(1)
# np.random.seed(1)


def main():
    if args.tensorboard: configure("runs/%s" % (args.name))

    # Data loading code
    normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.in_dataset == "CIFAR-10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../../data', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../../data', train=True, download=True,
                              transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        num_classes = 100

    if args.ood:
        ood_loader = torch.utils.data.DataLoader(
            TinyImages(transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
            batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    # create model
    model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)

    if args.adv:
        attack_in = LinfPGDAttack(model=model, eps=args.epsilon, nb_iter=args.iters, eps_iter=args.iter_size,
                                  rand_init=True, loss_func='CE')
        if args.ood:
            attack_out = LinfPGDAttack(model=model, eps=args.epsilon, nb_iter=args.iters, eps_iter=args.iter_size,
                                       rand_init=True, loss_func='OE')

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.ood:
        ood_criterion = OELoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    if args.lr_scheduler != 'cosine_annealing' and args.lr_scheduler != 'step_decay':
        assert False, 'Not supported lr_scheduler {}'.format(args.lr_scheduler)

    if args.lr_scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr))
    else:
        scheduler = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_scheduler == 'step_decay':
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if args.ood:
            if args.adv:
                train_ood(train_loader, ood_loader, model, criterion, ood_criterion, optimizer, scheduler, epoch,
                          attack_in, attack_out)
            else:
                train_ood(train_loader, ood_loader, model, criterion, ood_criterion, optimizer, scheduler, epoch)
        else:
            if args.adv:
                train(train_loader, model, criterion, optimizer, scheduler, epoch, attack_in)
            else:
                train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, attack_in=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()

    adv_losses = AverageMeter()
    adv_top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        # print(target)
        # print(target.size())
        # add_num = int(target.size()[0] / 7)
        # print(add_num)

        # print(type(input))
        # print(input.size())
        nat_input = input.detach().clone()

        # print(type(nat_input))
        # print(nat_input.size())

        labels = torch.full(size=(input.size()[0], 10), fill_value=0).cuda()
        labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)
        # print(labels)
        # add gaussian
        num_gaussian_inputs = input.size()[0] // 10
        # print(input.size())
        # print(labels.size())
        # 均一噪声
        uniform_input = torch.rand((num_gaussian_inputs, input.size()[1], input.size()[2], input.size()[3]))
        # uniform_labels = torch.full(size=(num_gaussian_inputs, 10), fill_value=0.1).cuda()
        # 标准高斯噪声
        gaussian_input = torch.randn((num_gaussian_inputs, input.size()[1], input.size()[2], input.size()[3]))
        gaussian_labels = torch.full(size=(num_gaussian_inputs, 10), fill_value=0.1).cuda()
        # 纯色图像
        # pure_input = torch.ones((num_gaussian_inputs, input.size()[1], input.size()[2], input.size()[3]))
        # ratio = torch.linspace(i / len(train_loader), (i + 1) / len(train_loader), steps=num_gaussian_inputs)
        # pure_input *= ratio.reshape(num_gaussian_inputs, 1, 1, 1)

        cat_input = torch.cat((nat_input, gaussian_input), 0)
        cat_labels = torch.cat((labels, gaussian_labels), 0)
        cat_input = torch.cat((nat_input, uniform_input), 0)
        cat_labels = torch.cat((labels, gaussian_labels), 0)  # 标签一样
        # 加了纯色好像变差
        # cat_input = torch.cat((nat_input, pure_input), 0)
        # cat_labels = torch.cat((labels, gaussian_labels), 0)  # 标签一样

        # print(nat_input.size())
        # print(labels.size())
        # exit(0)

        output = model(cat_input)
        # print(nat_output)
        # loss = criterion(output, target)
        # print(loss)

        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        nat_loss = -torch.sum(log_prob * cat_labels) / args.batch_size
        # print(nat_loss)
        # exit(0)

        # measure accuracy and record loss
        nat_prec1 = accuracy(output.data[:nat_input.size()[0]], target, topk=(1,))[0]
        nat_losses.update(nat_loss.data, input.size(0))
        nat_top1.update(nat_prec1, input.size(0))

        if not args.adv:
            # compute gradient and do SGD step
            loss = nat_loss
            if args.lr_scheduler == 'cosine_annealing':
                scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=nat_losses, top1=nat_top1))

            # gaussian训练

            # num_gaussian_inputs = input.size()[0] // 10
            # # print(input.size())
            # gaussian_input = torch.rand((num_gaussian_inputs, input.size()[1], input.size()[2], input.size()[3]))
            # nat_output = model(gaussian_input)
            # # print(nat_output)
            # # nat_loss = criterion(nat_output, target)
            # # print(nat_loss)
            #
            # labels = torch.full(size=(num_gaussian_inputs, 10), fill_value=0.1).cuda()
            # # print(labels)
            # log_prob = torch.nn.functional.log_softmax(nat_output, dim=1)
            # nat_loss = -torch.sum(log_prob * labels) / args.batch_size
            # loss = nat_loss
            # # print(loss)
            # # exit(0)
            # if args.lr_scheduler == 'cosine_annealing':
            #     scheduler.step()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()
            #
            # if i % args.print_freq == 0:
            #     # print('Epoch gaussian: [{0}][{1}/{2}]\t'
            #     #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #     #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #     #     epoch, i, len(train_loader), batch_time=batch_time,
            #     #     loss=nat_losses, top1=nat_top1))
            #     print(nat_output[0])
        else:
            adv_input = attack_in.perturb(input, target)
            adv_output = model(adv_input)
            adv_loss = criterion(adv_output, target)

            # measure accuracy and record loss
            adv_prec1 = accuracy(adv_output.data, target, topk=(1,))[0]
            adv_losses.update(adv_loss.data, input.size(0))
            adv_top1.update(adv_prec1, input.size(0))

            # compute gradient and do SGD step
            loss = adv_loss

            if args.lr_scheduler == 'cosine_annealing':
                scheduler.step()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Nat Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                      'Nat Prec@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                      'Adv Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                      'Adv Prec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    nat_loss=nat_losses, nat_top1=nat_top1, adv_loss=adv_losses, adv_top1=adv_top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('nat_train_loss', nat_losses.avg, epoch)
        log_value('nat_train_acc', nat_top1.avg, epoch)
        log_value('adv_train_loss', adv_losses.avg, epoch)
        log_value('adv_train_acc', adv_top1.avg, epoch)


def train_ood(train_loader_in, train_loader_out, model, criterion, ood_criterion, optimizer, scheduler, epoch,
              attack_in=None, attack_out=None):
    print('train in')
    kwargs = {'num_workers': 1, 'pin_memory': True}
    from utils.mixup_loader import MixupImages
    ood_loader = torch.utils.data.DataLoader(
        MixupImages(transform=transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
        batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    nat_in_losses = AverageMeter()
    nat_out_losses = AverageMeter()
    nat_top1 = AverageMeter()

    adv_in_losses = AverageMeter()
    adv_out_losses = AverageMeter()
    adv_top1 = AverageMeter()

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (in_set, mixup_set) in enumerate(zip(train_loader_in, ood_loader)):
        # print(in_set[0].size())
        # print(out_set[0].size())
        # print(mixup_set[0].size())
        # exit(0)

        in_len = len(in_set[0])  # 64
        out_len = len(mixup_set[0])  # 128
        # mixup_len = len(mixup_set[0])

        # num_gaussian_inputs = out_len // 8
        # 均一噪声
        # uniform_input = torch.rand((num_gaussian_inputs, 3, 32, 32))
        # uniform_labels = torch.full(size=(num_gaussian_inputs, 10), fill_value=0.1).cuda()
        # 标准高斯噪声
        # gaussian_input = torch.randn((num_gaussian_inputs, input.size()[1], input.size()[2], input.size()[3]))
        # gaussian_input = torch.randn((num_gaussian_inputs, 3, 32, 32))
        # 纯色图像
        # pure_input = torch.ones((num_gaussian_inputs, input.size()[1], input.size()[2], input.size()[3]))
        # ratio = torch.linspace(i / len(train_loader), (i + 1) / len(train_loader), steps=num_gaussian_inputs)
        # pure_input *= ratio.reshape(num_gaussian_inputs, 1, 1, 1)

        input = torch.cat((in_set[0], mixup_set[0]), 0)

        # out_set = torch.cat((out_set, ), 0)
        # cat_input = torch.cat((nat_input, uniform_input), 0)
        # out_len += mixup_len
        # print(in_len)
        # print(out_len)
        # exit(0)

        target = in_set[1]

        target = target.cuda()

        nat_input = input.detach().clone()
        nat_output = model(nat_input)

        nat_in_output = nat_output[:in_len]
        # nat_out_output = nat_output[in_len:]
        nat_out_output = nat_output[in_len:]
        nat_in_loss = criterion(nat_in_output, target)

        nat_out_loss = ood_criterion(nat_out_output)

        # measure accuracy and record loss
        nat_prec1 = accuracy(nat_in_output.data, target, topk=(1,))[0]
        nat_in_losses.update(nat_in_loss.data, in_len)
        nat_out_losses.update(nat_out_loss.data, out_len)
        nat_top1.update(nat_prec1, in_len)

        if not args.adv:
            # compute gradient and do SGD step
            loss = nat_in_loss + args.beta2 * nat_out_loss

            if args.lr_scheduler == 'cosine_annealing':
                scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                      'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader_in), batch_time=batch_time,
                    in_loss=nat_in_losses, out_loss=nat_out_losses, top1=nat_top1))
        else:
            adv_in_input = attack_in.perturb(in_set[0], target)

            adv_out_input = attack_out.perturb(out_set[0])

            adv_input = torch.cat((adv_in_input, adv_out_input), 0)
            adv_output = model(adv_input)

            adv_in_output = adv_output[:in_len]
            adv_out_output = adv_output[in_len:]

            adv_in_loss = criterion(adv_in_output, target)

            adv_out_loss = ood_criterion(adv_out_output)

            # measure accuracy and record loss
            adv_prec1 = accuracy(adv_in_output.data, target, topk=(1,))[0]
            adv_in_losses.update(adv_in_loss.data, in_len)
            adv_out_losses.update(adv_out_loss.data, out_len)
            adv_top1.update(adv_prec1, in_len)

            # compute gradient and do SGD step
            if args.adv_only_in:
                loss = adv_in_loss + args.beta2 * nat_out_loss
            else:
                loss = adv_in_loss + args.beta3 * adv_out_loss

            if args.lr_scheduler == 'cosine_annealing':
                scheduler.step()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Nat In Loss {nat_in_loss.val:.4f} ({nat_in_loss.avg:.4f})\t'
                      'Nat Out Loss {nat_out_loss.val:.4f} ({nat_out_loss.avg:.4f})\t'
                      'Nat Prec@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                      'Adv In Loss {adv_in_loss.val:.4f} ({adv_in_loss.avg:.4f})\t'
                      'Adv Out Loss {adv_out_loss.val:.4f} ({adv_out_loss.avg:.4f})\t'
                      'Adv Prec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})'.format(
                    epoch, i, len(train_loader_in), batch_time=batch_time,
                    nat_in_loss=nat_in_losses, nat_out_loss=nat_out_losses, nat_top1=nat_top1,
                    adv_in_loss=adv_in_losses, adv_out_loss=adv_out_losses, adv_top1=adv_top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('nat_train_loss', nat_losses.avg, epoch)
        log_value('nat_train_acc', nat_top1.avg, epoch)
        log_value('adv_train_loss', adv_losses.avg, epoch)
        log_value('adv_train_acc', adv_top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)


class OELoss(nn.Module):
    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, x):
        return -(x.mean(1) - torch.logsumexp(x, dim=1)).mean()


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


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= 60:
        lr *= 0.1
    if epoch >= 120:
        lr *= 0.1
    if epoch >= 160:
        lr *= 0.1
    # lr = args.lr * (0.1 ** (epoch // 60)) * (0.1 ** (epoch // 80))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    main()
