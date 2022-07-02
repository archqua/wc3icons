from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms as tt
import random
import matplotlib.pyplot as plt
import numpy as np


def load_as_tensor(dir, crop=4):
    size = (64 - 2*crop,) * 2
    files = os.listdir(dir)
    tensor = torch.empty((len(files), 3, *size), dtype=torch.float32)
    for i, file in enumerate(files):
        try:
            with Image.open(os.path.join(dir, file)) as img:
                if img.size[0] != 3:
                    bg = Image.new("RGB", img.size, (0,0,0))
                    bg.paste(img)
                    img = bg
                tensor[i] = tt.CenterCrop(size)(tt.ToTensor()(img))
        except:
            print(f"{file}: {img.size}")
            # return None
    return tensor

def darken_edge(pic):
    while len(pic.shape) < 4:
        pic = pic[None,:]
    # darken top and bottom
    for j in range(2):
        factor = 1/3 + j/2
        pic[:,:,:,j] *= factor
        pic[:,:,:,-j-1] *= factor
    # darken left and right
    for i in range(2):
        factor = 1/3 + i/2
        pic[:,:,i,:] *= factor
        pic[:,:,-i-1,:] *= factor
    pic = pic.squeeze()
    return pic

def draw(pics, titles=None):
    count = pics.shape[0]
    fig, axes = plt.subplots(1, count, sharey=True, figsize=(3*count,3))
    # titles = [""] + ["train"] + [""]*2 + ["eval"] + [""]
    for i, pic in enumerate(pics):
        pic = pic.detach().cpu().squeeze().numpy()
        axes[i].imshow(np.rollaxis(pic, 0, 3))
        axes[i].axis('off')
        if titles is not None:
          axes[i].set_title(titles[i])


class PhotoLoader:
    def __init__(self, photos, transforms=tt.Compose([tt.RandomHorizontalFlip()])):
        self.photos = photos
        self.transforms = transforms
        self.i = 0

    def __len__(self):
        return len(self.photos)

    def get_batch(self, batch_size):
        last_i = self.i + batch_size
        if last_i <= len(self.photos):
            res = self.photos[self.i:self.i + batch_size]
        else:
            res1 = self.photos[self.i:]
            last_i %= len(self.photos)
            res2 = self.photos[:last_i]
            res = torch.concat((res1, res2))
        self.i += batch_size
        self.i %= len(self.photos)
        if self.transforms is not None:
            for i in range(len(res)):
                res[i] = self.transforms(res[i])
        return res


class TestLoader:
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return len(self.x)

    def get_batch(self, batch_size):
        idx = np.random.randint(0, len(self.x), batch_size)
        return self.x[idx]


class DiscriminationBuffer:
    def __init__(self, n_items):
        self.n_items = n_items
        self.items = []

    def put(self, item):
        if len(self.items) < self.n_items:
            self.items.append(item)
        else:
            del self.items[0]
            self.put(item)

    def get(self, i=None):
        if i is None:
            i = random.randint(0, len(self.items) - 1)
            return self.get(i)
        else:
            return self.items[i]

