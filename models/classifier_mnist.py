from traitlets.traitlets import List
import torch

from torch import nn
from torch.nn import functional as F

class ClassifierMNIST(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_outputs: int=10,
                 hidden_dims: List =None,
                 **kwargs) -> None:
        super(ClassifierMNIST, self).__init__()

        modules=[]
        for num in range(3):
            if num==0:
                modules.append(
                    nn.Sequential(
                        nn.Linear(784,1024),
                        nn.BatchNorm1d(1024),
                        nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(1024,1024),
                        nn.BatchNorm1d(1024),
                        nn.LeakyReLU())
                )
        self.encoder = nn.Sequential(*modules)
        self.linear = nn.Linear(1024,num_outputs)
        self.relu = nn.LeakyReLU()

    
    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)
        out = self.encoder(x)
        out = self.linear(out)
        out = self.relu(out)
        
        return out
        
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the cross entropy loss function.
        
        """
        scores=args[0]
        y=args[1]
        loss = F.cross_entropy(scores, y)
        
        return {'loss': loss}
