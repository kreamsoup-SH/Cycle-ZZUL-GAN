import os
import glob
import random
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image


def read_image(image):
    image = Image.open(image)
    image = image.convert('L')
    return image

def omniglot_prototype_collate_fn(batched_data):
    support_images, query_images, query_labels = [], [], []
    for i in range(len(batched_data)):
        support_images.append(batched_data[i][0])
        query_images.append(batched_data[i][1])
        query_labels.append(torch.tensor([i,i,i,i,i], dtype=torch.long))
    # Fill this
    support_images = torch.stack(support_images, dim=0)
    query_images = torch.stack(query_images, dim=0)
    query_labels = torch.stack(query_labels, dim=0)
    #print(f'support_images의 크기 : {support_images.shape}')
    return support_images, query_images, query_labels


class OmniglotBaseline(Dataset):
    def __init__(self, root, K_s=5, K_q=1, training=False, transform=None):
        super().__init__()
        self.data = []
        for character in os.listdir(root):
            images = glob.glob(f'{root}/{character}/*')
            images = map(read_image, images)
            images = map(to_tensor, images)
            images = torch.stack(list(images), dim=0)
            self.data.append(images)
        self.transform = transform
        self.training = training
        self.K_s = K_s
        self.K_q = K_q
        if training:
            self.image_idx = [i for i in range(K_s)]
        else:
            self.image_idx = [i for i in range(K_s, K_s + K_q)]
    
    def __getitem__(self, idx):
        K = len(self.image_idx)
        images = self.data[idx][self.image_idx, ...]
        if self.transform is not None:
            for k in range(K):
                images[k, ...] = self.transform(images[k, ...])
        labels = idx * torch.ones((K,), dtype=torch.long)
        return images, labels
    
    def __len__(self):
        return len(self.data)


class OmniglotPrototype(Dataset):
    def __init__(self, root, K_s=5, K_q=1, training=False, transform=None):
        super().__init__()
        self.data = []
        for character in os.listdir(root):
            images = glob.glob(f'{root}/{character}/*')
            images = map(read_image, images)
            images = map(to_tensor, images)
            images = torch.stack(list(images), dim=0)
            self.data.append(images)
        
        self.K_s = K_s
        self.K_q = K_q
        self.transform = transform
        self.image_idx = [i for i in range(20)]
        self.training = training
    
    def __getitem__(self, idx):
        K = self.K_s + self.K_q   
        if self.training:
            image_idx = random.sample(self.image_idx, k=K)
            images = self.data[idx][image_idx, ...]
        else:
            images = self.data[idx][:K, ...]

        if self.transform is not None:
            for k in range(K):
                images[k, ...] = self.transform(images[k, ...])
        
        support_images, query_images = images.split([self.K_s, self.K_q], dim=0)
        query_labels = idx * torch.ones((self.K_q,), dtype=torch.long)
        return support_images, query_images, query_labels
    
    def __len__(self):
        return len(self.data)

