from traitlets.traitlets import List
import torch

from torch import nn
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_outputs: int=10,
                 hidden_dims: List =None,
                 **kwargs) -> None:
        super(Classifier, self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]
        self.hidden_dims = hidden_dims
        
        #conv layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        #affine layer
        self.fc = nn.Sequential(
                        nn.LeakyReLU(),
                        nn.Linear(hidden_dims[-1], num_outputs),
                    )
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
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
