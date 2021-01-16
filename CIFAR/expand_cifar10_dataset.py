import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class expandedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(expandedCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.one_hot_like_labels = np.eye(10)[self.targets]

        expand_data = []
        expand_targets = []

        # now load the picked numpy arrays
        for i in range(4):
            images = np.load(f'images{i+2}.npy')
            labels = np.load(f'labels{i+2}.npy')
            expand_data.append(images)
            expand_targets.append(labels)

        expand_data = np.vstack(expand_data).reshape(-1, 3, 32, 32)
        expand_data = expand_data.transpose((0, 2, 3, 1))
        expand_targets = np.vstack(expand_targets)
        # print(expand_data.shape)
        # print(expand_targets)
        self.data = np.concatenate((self.data, expand_data))
        self.one_hot_like_labels = np.concatenate((self.one_hot_like_labels, expand_targets))






normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ToPILImage()
])

topil = transforms.ToPILImage()

batch_size = 8

cifar10dataset = expandedCIFAR10('../../data', train=True, download=True,
                                  transform=transform_train)

print(cifar10dataset.data.shape)
print(len(cifar10dataset.one_hot_like_labels))