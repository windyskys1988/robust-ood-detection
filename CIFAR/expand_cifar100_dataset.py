import torchvision.datasets as datasets
import numpy as np
import torch

# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import torchvision
# from typing import Tuple, Any
# import torchvision.transforms as transforms
# from torch.utils.data import SubsetRandomSampler

class taggedCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False,num=10,index=0,tag=np.array(range(100)),random=False):
        # if random:
        #     indices = list(range(len(100)))
        #     np.random.shuffle(indices)
        #     tag = indices[0:num]
        # else:
        #     tag = np.array(range(index,num+index))
        tag_init=np.array(range(len(tag)))
        tagmap=dict(zip(tag,tag_init))
        super(taggedCIFAR100, self).__init__(root, transform=transform,train=train,
                                              target_transform=target_transform, download=download)
        mask = [i for i in range(len(self.targets)) if (self.targets[i] in tag)]
        targetss = [tagmap[self.targets[i]] for i in range(len(self.targets)) if (self.targets[i] in tag)]
        class_to_idx=dict()
        for i in range(len(tag)):
            class_to_idx[self.classes[tag[i]]]=i
        classes = list(class_to_idx.keys())
        #mask_for_one_hot = [i for i in range(len(self.one_hot_like_labels)) if
                            #(self.one_hot_like_labels[i] in one_hot_tag)]
        # print(mask_for_one_hot)
        mask_tensor=torch.Tensor(mask)
        # mask_for_one_hot=torch.from_numpy(mask_for_one_hot)
        # self.data= torch.index_select(self.data,0,mask_tensor)
        # self.one_hot_like_labels=torch.index_select(self.one_hot_like_labels, 0, mask_tensor)
        self.data = self.data[mask, :]
        self.targets = targetss
        self.classes=classes
        self.class_to_idx=class_to_idx


# normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                   std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
#
# transform_train = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.ToPILImage()
# ])
# topil = transforms.ToPILImage()
# batch_size = 16
# tagstr ='5,6,7,8,9'
# tags = np.array(tagstr.split(',')).astype(np.int)
# taggedcifar100 = taggedCIFAR100('../../data', train=True, download=True,
#                                  transform=transform_train,tag=np.array((range(10,20,1))))
#
# indices = list(range(len(taggedcifar100)))
# np.random.shuffle(indices)
# sampler = SubsetRandomSampler(indices)
# train_loader = torch.utils.data.DataLoader(taggedcifar100, batch_size=batch_size,
#     sampler=sampler)
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# images = images.numpy()  # convert images to numpy for display
#
# # 显示图像，标题为类名
# fig = plt.figure(figsize=(10, 10))
# # 显示16张图片
# for idx in np.arange(batch_size):
#     ax = fig.add_subplot(4, batch_size / 4, idx + 1, xticks=[], yticks=[])
#     img = images[idx]
#     plt.imshow((np.transpose(img, (1, 2, 0))))
#     tag = labels[idx].numpy()
#     ax.set_title(taggedcifar100.classes[tag])
# fig.savefig("output2.jpg")
# print(taggedcifar100.data.shape)
# print(taggedcifar100.targets)