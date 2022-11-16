import argparse
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchmetrics.functional.classification import accuracy

from src.datasets import OmniglotPrototype, omniglot_prototype_collate_fn
from src.models import PrototypeNet
from src.engines import evaluate_prototype

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="prototype")
parser.add_argument("--device", type=str, default="cuda:3")
parser.add_argument("--root", type=str, default="data/omniglot/meta-test")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--alphabets", type=str, nargs=5, default=["Atlantean", "Japanese_(hiragana)", "Japanese_(katakana)", "Korean", "ULOG"])
parser.add_argument("--num_characters", type=int, nargs=5, default=[26, 52, 47, 40, 26])
parser.add_argument("--num_supports", type=int, default=5)
parser.add_argument("--num_queries", type=int, default=5)
parser.add_argument("--checkpoints", type=str, default='checkpoints')
args = parser.parse_args()


def main(args):
    accuracies = []

    for alphabet, num_classes in zip(args.alphabets, args.num_characters):
        # Build dataset
        root = f'{args.root}/{alphabet}'
        dataset = OmniglotPrototype(root, K_s=args.num_supports, K_q=args.num_queries, training=False, transform=None)
        loader = DataLoader(dataset, num_classes, num_workers=args.num_workers, collate_fn=omniglot_prototype_collate_fn)

        # Build model
        model = PrototypeNet()
        state_dict = torch.load(f'{args.checkpoints}/{args.title}.pth')
        model.load_state_dict(state_dict['model'])
        model = model.to(args.device)

        # Build loss and metric
        loss_fn = nn.CrossEntropyLoss()
        metric_fn = accuracy

        # Main loop
        summary = evaluate_prototype(loader, model, loss_fn, metric_fn, args.device)
        accuracies.append(summary["metric"])
    
    # Print performance
    for i, alphabet in enumerate(args.alphabets):
        print(f'{alphabet}: {accuracies[i]:.4f}')
    mean_accuracy = np.mean(accuracies)
    mean_std = np.std(accuracies)
    print(f'mean: {mean_accuracy:.4f}, std: {mean_std:.4f}')



if __name__=="__main__":
    main(args)