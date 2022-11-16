import argparse
import functools
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchmetrics.functional.classification import accuracy

from src.datasets import OmniglotPrototype, omniglot_prototype_collate_fn
from src.models import PrototypeNet
from src.engines import train_prototype
from src.utils import save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="prototype")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--root", type=str, default="data2/omniglot/meta-train")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--num_supports", type=int, default=5)
parser.add_argument("--num_queries", type=int, default=5)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--pretrain", type=bool, default=True)
args = parser.parse_args()


def main(args):
    # Build dataset
    transform = T.RandomCrop((32, 32), padding=4)
    dataset = OmniglotPrototype(args.root, K_s=args.num_supports, K_q=args.num_queries, training=True, transform=transform)
    loader = DataLoader(dataset, args.num_classes, shuffle=True, num_workers=args.num_workers, drop_last=True, 
                        collate_fn=omniglot_prototype_collate_fn)

    # Build model
    model = PrototypeNet()
    model = model.to(args.device)

    # Build optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy

    # Main loop
    for epoch in range(args.epochs):
        summary = train_prototype(loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
        print(f'Epoch: {epoch + 1}, Accuracy: {summary["metric"]:.4f}')
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)


if __name__=="__main__":
    main(args)