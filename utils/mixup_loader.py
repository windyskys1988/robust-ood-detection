import numpy as np
import torch
from PIL import Image

class MixupImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=True):
        self.transform = transform
        expand_data = []

        # now load the picked numpy arrays
        for i in range(1):
            images = np.load(f'images_2_{i + 2}.npy')
            expand_data.append(images)

        expand_data = np.vstack(expand_data).reshape(-1, 3, 32, 32)
        expand_data = expand_data.transpose((0, 2, 3, 1))
        # print(expand_data.shape)
        self.data = expand_data



    def __getitem__(self, index):
        img = self.data[index]

        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


    def __len__(self):
        return self.data.shape[0]
