import glob
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image


def read_image(image):
    image = Image.open(image)
    image = image.convert('RGBA').convert('RGB')
    image = image.resize((100 , 100))
    return image

class CustomDataset(Dataset):
    def __init__(self, _dir, _name):
        super().__init__()
        print(f'start_name={_name}')
        self.data=[]
        data = glob.glob(f'{_dir}/{_name}/*')
        data = map(read_image,data)
        data = map(to_tensor, data)
        data = torch.stack(list(data), dim=0)
        self.data=data
        print(f'end_name={_name}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
