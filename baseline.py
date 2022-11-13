import argparse

from src.datasets import CustomDataset
from torch.utils.data import DataLoader, random_split

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="baseline")
parser.add_argument("--device", type=str, default="cuda:3")
parser.add_argument("--data", type=str, default="data")
parser.add_argument("--cons_root", type=str, default="cons")
parser.add_argument("--con_name", type=str, default="zzul")
parser.add_argument("--imgs_root", type=str, default="images")
parser.add_argument("--img_name", type=str, default="celeba")
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

def main(args):
    # Build Dataset
    cons_dir = f'{args.data}/{args.cons_root}'
    imgs_dir = f'{args.data}/{args.imgs_root}'
    input1 = CustomDataset(cons_dir, args.con_name)
    input2 = CustomDataset(imgs_dir, args.img_name)

    input1_size = input1.__len__()
    print(input1_size)
    input2_size = input2.__len__()
    print(input2_size)
    train1_size = int(input1_size * 0.8)
    train2_size = int(input2_size * 0.8)

    train_input1, test_input1 = random_split(input1, [train1_size, input1_size-train1_size])
    train_input2, test_input2 = random_split(input2, [train2_size, input2_size-train2_size])

    print('test')
    print(train_input1.__len__())
    print(test_input1.__len__())


if __name__=='__main__':
    main(args)