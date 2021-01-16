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
import matplotlib.pyplot as plt
from PIL import Image

normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ToPILImage()
])

topil = transforms.ToPILImage()

batch_size = 8

cifar10dataset = datasets.CIFAR10('../../data', train=True, download=True,
                                  transform=transform_train)
# print(cifar10dataset.data.shape)
# print(len(cifar10dataset.targets))

eye = np.eye(10)

# 分别获取标签为0~9的样本的位置indext
sample_positions = []
for i in range(10):
    indices = np.nonzero(np.array(cifar10dataset.targets) == i)
    sample_positions.append(indices)

# print(sample_positions)
sample_positions = np.array(sample_positions).squeeze()

# print(sample_positions.shape)

# c1_pos = sample_positions[0][0:8]
# c2_pos = sample_positions[1][0:8]
# # print(c1_pos)
# c1_images = cifar10dataset.data[c1_pos].transpose(0, 3, 1, 2)
# # print(torch.from_numpy(c1_images).size())
# c2_images = cifar10dataset.data[c2_pos].transpose(0, 3, 1, 2)
# avg = (c1_images + c2_images) /2
# print(np.max(c1_images), np.max(c2_images), np.max(avg))
# images = np.vstack((c1_images, c2_images, avg))/255
# trans_images = torchvision.utils.make_grid(torch.from_numpy(images), nrow=batch_size).numpy().transpose(1, 2, 0)
# print(trans_images.shape)
# plt.imshow(trans_images)
# plt.show()
count = 5000

np.random.seed(1)


def combine_images(sample_positions, num_classes=2, num_images=count):
    perm = [np.random.permutation(count) for _ in range(num_classes)]
    perm = np.array(perm)[:, 0:num_images]
    # print(perm)
    # print(sample_positions[0:num_classes, perm])
    images = []
    labels = []
    for i in range(num_classes):
        # print(perm[i])
        images.append(cifar10dataset.data[sample_positions[i][perm[i]]])
        # print(eye[i].expand_dims(1).shape)
        labels.append(np.expand_dims(eye[i], axis=0).repeat(num_images, axis=0))

    # return np.vstack(images)
    return np.array(images), np.array(labels)


for i in range(4):
    images, labels = combine_images(sample_positions, num_classes=i+2, num_images=5000)
    print(images.shape)
    print(labels.shape)
    fused_images = np.mean(images, axis=0)
    fused_labels = np.mean(labels, axis=0)

    np.save(f'images{i+2}.npy', fused_images)
    np.save(f'labels{i+2}.npy', fused_labels)

    print(fused_images.shape)
    print(fused_labels.shape)
