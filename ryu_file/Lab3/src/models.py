import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_dim, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Block(1, 32, 3, 2, 1),
            Block(32, 64, 3, 2, 1),
            Block(64, 128, 3, 2, 1),
            Block(128, 256, 3, 1, 1),
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.mean([-1, -2])
        return x

class BaselineNet(nn.Module):
    def __init__(self, num_classes=1432, pretrain=False):
        super().__init__()
        self.features = EmbeddingNet()
        self.head = nn.Linear(256, num_classes)
        self.pretrain = pretrain
    
    def forward(self, x):
        if not self.pretrain:
            N, K, C, H, W = x.shape
            x = x.reshape((N * K, C, H, W))
        print(f'embeddin전 x.shape : {x.shape}')
        x = self.features(x)
        print(f'embeddin후 x.shape : {x.shape}')
        x = self.head(x)
        print(f'linear후 x.shape : {x.shape}')
        if not self.pretrain:
            x = x.reshape((N, K, -1))
        return x


class PrototypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = EmbeddingNet()
    
    def forward(self, x_s, x_q):
        # x_s.shape = [20, 5, 1, 32, 32]
        # x_q.shape = [20, 5, 1, 32, 32]
        N_s, K_s, C_s, H_s, W_s = x_s.shape
        N_q, K_q, C_q, H_q, W_q = x_q.shape

        x_s = x_s.reshape((N_s * K_s, C_s, H_s, W_s)) # torch.Size(20 * 5, 1, 32, 32)
        x_q = x_q.reshape((N_q * K_q, C_q, H_q, W_q)) # torch.Size(20 * 5, 1, 32, 32)
        # x_s.shape = [100, 1, 32, 32]
        # x_q.shape = [100, 1, 32, 32]

        z_s = self.features(x_s)
        z_q = self.features(x_q)
        # z_s.shape = [100, 256]
        # z_q.shape = [100, 256]
        
        c_s = self.compute_prototype(z_s)
        # c_s.shape = [20, 256]

        d = self.compute_distance(z_q, c_s)      
        y_q = F.softmax(d, dim=0)
        #print(f'y_q의 크기 : {y_q.shape}') #torch.Size([20,5])
        #print(f'torch.sum 의 크기 : {torch.sum(y_q[:,0])}') #1.0
        y_q = y_q.transpose(1,0) #torch.Size(5,20)
        return y_q
        
    def compute_prototype(self, z_s):
        z_s = z_s.reshape((20, 5, -1))
        c_s = z_s.mean(dim=1)
        return c_s

    def compute_distance(self, z_q, c_s):
        # z_q = torch.Size([100, 256])
        # c_s = torch.Size([20, 256])
        #z_q = z_q.reshape((20, 5, -1)) # z_q = torch.Size([20, 5, 256])
        #z_q = z_q.transpose(1,0) # z_q = torch.Size([5, 20, 256])
        #c_s = c_s.reshape((1, 20, -1)) # c_s = torch.Size([1, 20, 256])
        d = []
        #print(f'계산 결과 : {((z_q[0,:] - c_s[0,:])**2).mean()}')
        for i in range(100):
            for j in range(20):
                d.append(((z_q[i,:] - c_s[j,:])**2).mean())
        d = torch.tensor(d)
        #print(f'd 의 크기 : {d.shape}') # d = torch.Size([20, 5])
        d = d.reshape((5,20,20))
        return d    

class FixedPrototypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = EmbeddingNet()
    
    def forward(self, x_s, x_q):
        z_s = self.extract_features(x_s)
        z_q = self.extract_features(x_q)
        c_s = self.compute_prototype(z_s)
        y_q = -self.compute_distance(z_q, c_s)
    def extract_features(self, x):
        N, K, C, H, W = x.shape
        x = x.reshape((N * K, C, H, W))
        x = self.features(x)
        x = x.reshape((N,K,-1))
        return x
        
    def compute_prototype(self, z_s):
        c_s = z_s.mean(dim=1)
        return c_s

    def compute_distance(self, z_q, c_s):
        N, K, D = z_q.shape
        z_q = z_q.reshape((N, K, 1, D))
        c_s = c_s.reshape((1, 1, N, D))
        d = ((z_q - c_s) ** 2).sum(dim=-1)
        return d    