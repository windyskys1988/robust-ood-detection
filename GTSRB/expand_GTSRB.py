import torchvision.datasets as datasets
import numpy as np
import torch
import os

import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision
from typing import Tuple, Any
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

import ipdb

class taggedGTSRB(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None,tag=np.array(range(10))):
        # if random:
        #     indices = list(range(len(100)))
        #     np.random.shuffle(indices)
        #     tag = indices[0:num]
        # else:
        #     tag = np.array(range(index,num+index))
        tag_init=np.array(range(len(tag)))
        tagmap=dict(zip(tag,tag_init))
        if train:
            root = os.path.join(root, 'train')
        else:
            root=os.path.join(root,'test')
        super(taggedGTSRB, self).__init__(root, transform=transform)
        imgs = [(self.imgs[i][0],tagmap[self.imgs[i][1]]) for i in range(len(self.targets)) if (self.targets[i] in tag)]
        targetss = [tagmap[self.targets[i]] for i in range(len(self.targets)) if (self.targets[i] in tag)]
        class_to_idx=dict()
        for i in range(len(tag)):
            class_to_idx[self.classes[tag[i]]]=i
        classes = list(class_to_idx.keys())
        #mask_for_one_hot = [i for i in range(len(self.one_hot_like_labels)) if
                            #(self.one_hot_like_labels[i] in one_hot_tag)]
        # print(mask_for_one_hot)
        # mask_for_one_hot=torch.from_numpy(mask_for_one_hot)
        # self.data= torch.index_select(self.data,0,mask_tensor)
        # self.one_hot_like_labels=torch.index_select(self.one_hot_like_labels, 0, mask_tensor)
        self.samples= imgs
        self.imgs = imgs
        self.targets = targetss
        self.classes=classes
        self.class_to_idx=class_to_idx

#
# normalizer = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                   std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
#
# transform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomCrop(32),
#     transforms.ToTensor(),
# ])
# topil = transforms.ToPILImage()
# batch_size = 16
# tagstr ='30,31,32,33,34'
# tags = np.array(tagstr.split(',')).astype(np.int)
# datas = taggedGTSRB(root='./datasets/gtsrb/data',train=True,transform=transform,tag=tags)
# indices = list(range(int(len(datas)*0.8)))
# np.random.shuffle(indices)
# sampler = SubsetRandomSampler(indices[30:60])
# train_loader = torch.utils.data.DataLoader(taggedGTSRB(root='./datasets/gtsrb/data',train=True,transform=transform,tag=tags), batch_size=batch_size,sampler=sampler)
# for i, data in enumerate(train_loader):
#     input, target = data
#     print(len(input))
#     print(target)
#
# # dataiter = iter(train_loader)
# # images, labels = dataiter.next()
# # images = images.numpy()  # convert images to numpy for display
# # 显示图像，标题为类名
# # fig = plt.figure(figsize=(10, 10))
# # # 显示16张图片
# # for idx in np.arange(batch_size):
# #     ax = fig.add_subplot(4, batch_size / 4, idx + 1, xticks=[], yticks=[])
# #     img = images[idx]
# #     plt.imshow((np.transpose(img, (1, 2, 0))))
# #     tag = labels[idx].numpy()
# #     ax.set_title(tag)
# # fig.savefig("GTSRB.jpg")
# # print(len(datas.targets))
# # print(labels)