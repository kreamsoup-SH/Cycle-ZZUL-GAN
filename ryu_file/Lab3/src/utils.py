import os
import torch

def save_pretrained_embeddingnet(checkpoint_dir, title, model):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f'{checkpoint_dir}/{title}_embedding.pth'
    torch.save(model.state_dict(), checkpoint_path)

def save_checkpoint(checkpoint_dir, title, model, optimizer, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(state_dict, checkpoint_path)