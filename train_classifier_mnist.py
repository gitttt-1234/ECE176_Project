import torch
from models import *
from models import ClassifierMNIST,VanillaVAEMNIST,Actor,BaseVAE
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import torchvision.transforms as T
import yaml
from pytorch_lightning.utilities.seed import seed_everything
from dataset_mnist import VAEDataset
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
import torchvision.utils as vutils
import numpy as np


def check_accuracy(loader, model):
    device = torch.device('cuda')
    n_corr = 0
    n_samples = 0
    model.eval()  
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            n_corr += (preds == y).sum()
            n_samples += preds.size(0)
        acc = float(n_corr) / n_samples
        print('Accuracy (%.2f)' % (100 * acc))
        return acc

def train_classifier(model, optimizer, epochs,exp_flag,latent_dim,vaemodel,acmodel):
    device = torch.device('cuda')
    model = model.to(device=device)
    accuracies = []
    for i in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)
            if(exp_flag==1):
                    bs = x.shape[0]
                    random_indices_img_z = torch.randint(low=0, high=bs, size=(bs//5,))
                    vutils.save_image(x[random_indices_img_z],
                          os.path.join(log_dir, 
                                       "input_mnist", 
                                       f"input_Epoch_{i}.png"),
                          normalize=True,
                          nrow=12)
                    
                    with torch.no_grad():
                        fake_z = torch.randn(bs // 5, latent_dim)
                        fake_z = fake_z.to(device=device, dtype=dtype)
                            
                        labels = torch.eye(num_class)
                        labels = labels.to(device=device)
                        labels = labels[y[random_indices_img_z]]
                        z_g = acmodel(fake_z, labels)
                        img_gen = vaemodel.decode(z_g)
                    vutils.save_image(img_gen,
                          os.path.join(log_dir, 
                                       "recons_mnist", 
                                       f"recons_Epoch_{i}.png"),
                          normalize=True,
                          nrow=12)
                    x[random_indices_img_z] = img_gen
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

           
        print(('Epoch %d, loss = %.4f' % (i, loss.item())))
        accuracies.append(check_accuracy(loader_val, model))
        print()
    return accuracies           


#############################################################################################################


device = torch.device('cuda')
dtype = torch.float32

parser = argparse.ArgumentParser(description='Generic runner for classifer')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')


parser.add_argument('--vae-config',  '-vc',
                    dest="vfilename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
                    
parser.add_argument('--ac-config',  '-ac',
                    dest="acfilename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/ac.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
with open(args.vfilename, 'r') as file:
    try:
        vconfig = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

with open(args.acfilename, 'r') as file:
    try:
        acconfig = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

vaemodel = vae_models[vconfig['model_params']['name']](**vconfig['model_params'])
state_dict = torch.load(config['model_params']['vae_ckpt'])['state_dict']

for key in list(state_dict.keys()):
    state_dict[key.replace("model.","")] = state_dict.pop(key)
vaemodel.load_state_dict(state_dict, strict=True)
vaemodel = vaemodel.to(device=device)
vaemodel.eval()

acmodel = vae_models[acconfig['model_params']['actor_name']](**acconfig['model_params'])
state_dict = torch.load(config['model_params']['ac_ckpt'])['state_dict']

for key in list(state_dict.keys()):
    new_key = key.split('.',1)[1]
    state_dict[new_key] = state_dict.pop(key)

acmodel.load_state_dict(state_dict, strict=False)
acmodel = acmodel.to(device=device)
acmodel.eval()

latent_dim = acconfig['model_params']['latent_dim']
NUM_TRAIN = 60000
train_batch_size= config['data_params']['train_batch_size']
val_batch_size = config['data_params']['val_batch_size']
data_path = config['data_params']['data_path']
patch_size = config['data_params']['patch_size']
in_channels= config['model_params']['in_channels']
num_class = config['model_params']['n_class']
learning_rate = config['exp_params']['LR']
weight_decay = config['exp_params']['weight_decay']
epochs = 50

log_dir = config['logging_params']['save_dir']

train_transform = transform = T.Compose([
                T.Resize(patch_size),
                T.ToTensor(),
                T.Normalize((0.5), (0.25))
                
            ])
mnist_train = dset.MNIST(data_path, train=True, download=True,
                             transform=train_transform)
loader_train = DataLoader(mnist_train, batch_size=train_batch_size, num_workers=2,
                          shuffle=True)

mnist_test = dset.MNIST(data_path, train=False, download=True, 
                            transform=transform)
loader_val = DataLoader(mnist_test, batch_size=val_batch_size, num_workers=2)


Path(f"{log_dir}/input_mnist").mkdir(exist_ok=True, parents=True)
Path(f"{log_dir}/recons_mnist").mkdir(exist_ok=True, parents=True)
acc=[]
print("MNIST")
for exp_flag in range(0,2):
    if(exp_flag==0):
        print("---------------MNIST generic classifer---------------")
    else:
        print("---------------MNIST classifer with latent constraints---------------")
    model = ClassifierMNIST(in_channels=in_channels,num_outputs=num_class)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    acc.append(train_classifier(model,optimizer,epochs,exp_flag,latent_dim,vaemodel,acmodel))
    print("\n")
    
plt.figure()
plt.plot(range(1,epochs+1),acc[0],label='generic classifer')
plt.plot(range(1,epochs+1),acc[1],label='classifer with latent constraints')
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.legend()
plt.show()


                