
import torch
import torchvision
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import Subset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        # Images and labels are tensors. Images are already augmented and on GPU.
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return len(self.images)

def build_dataset(device="cuda"):
    _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])
    train_val_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_set = Subset(train_val_set, range(40000))
    val_set = Subset(train_val_set, range(40000, 50000))
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    
    def dataset_to_lists(dataset):
        # Given a dataset (train, val or test) return two lists: one with
        # images and one with labels
        images = []
        labels = []
        for i, l in dataset:
            images.append(i.unsqueeze(0))
            labels.append(l)
        images = torch.cat(images).to(device).half()
        labels = torch.Tensor(labels).long().to(device)
        return images, labels
    
    train_imgs, train_labels = dataset_to_lists(train_set)
    val_imgs, val_labels = dataset_to_lists(val_set)
    test_imgs, test_labels = dataset_to_lists(test_set)
    
    return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels

class RandomCrop(nn.Module):
    def __init__(self, size=32, pad=4):
        super(RandomCrop, self).__init__()
        self.size = size
        self.pad = pad
    def forward(self, x):
        i = torch.randint( 2 *self.pad, (2,)).to(x.device).long()
        return x[:, :, i[0]:i[0 ] +self.size, i[1]:i[1 ] +self.size]

class RandomHorizontalFlip(nn.Module):
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__()
    def forward(self, x):
        r = torch.randn((x.shape[0], 1, 1, 1), device=x.device) < 0.
        return r* x + (~r) * x.flip(-1)


class Cutout(nn.Module):
    def __init__(self, height, width):
        super(Cutout, self).__init__()
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.shape[2], image.shape[3]
        mask = np.ones((1, 1, h, w), np.float32)
        y = np.random.choice(range(h))
        x = np.random.choice(range(w))

        y1 = np.clip(y - self.height // 2, 0, h)
        y2 = np.clip(y + self.height // 2, 0, h)
        x1 = np.clip(x - self.width // 2, 0, w)
        x2 = np.clip(x + self.width // 2, 0, w)

        mask[:, :, y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype)
        mask = mask.expand_as(image)
        image *= mask
        return image


# run data augmentation as a module on gpu
class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()
        t = torch.nn.Sequential(
            transforms.RandomCrop(32, (4, 4)),
            transforms.RandomHorizontalFlip(),
            Cutout(8, 8)
        )
        self.transforms = t  # torch.jit.script(t)

    def forward(self, x):
        x = self.transforms(x)
        return x

