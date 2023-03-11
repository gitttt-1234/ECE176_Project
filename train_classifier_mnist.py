import torch
from models import *
from models import ClassifierMNIST,VanillaVAEMNIST,Actor,BaseVAE
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
from pytorch_lightning.loggers import TensorBoardLogger
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

def train_classifier(model, optimizer, epochs,exp_flag,latent_dim,vaemodel,acmodel):
    device = torch.device('cuda')
    model = model.to(device=device)  
    for i in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)
            if(exp_flag!=0):
                    bs = x.shape[0]
                    random_indices_img_z = torch.randint(low=0, high=bs, size=(bs//2,))
                    vutils.save_image(x[random_indices_img_z],
                          os.path.join(log_dir, 
                                       "input_mnist", 
                                       f"input_Epoch_{i}.png"),
                          normalize=True,
                          nrow=12)
                    if exp_flag == 1:
                        with torch.no_grad():
                            mu, var = vaemodel.encode(x[random_indices_img_z])
                            #noise = torch.randn(mu.shape)*0.1
                            noise = torch.normal(mean=0.0, std=0.1,size=mu.shape)
                            noise = noise.to(device=device, dtype=dtype)
                            real_z = vaemodel.reparameterize(mu+noise, var+noise)
                            img_gen = vaemodel.decode(real_z)
                    elif exp_flag == 2:  
                        with torch.no_grad():
                            fake_z = torch.randn(bs // 2, latent_dim)
                            fake_z = fake_z.to(device=device, dtype=dtype)
                            
                            labels = torch.eye(num_class)
                            labels = labels.to(device=device)
                            labels = labels[y[random_indices_img_z]]
                            
                             
                            
                            z_g = acmodel(fake_z, labels)
                            img_gen = vaemodel.decode(z_g)
                    vutils.save_image(img_gen,
                          os.path.join(log_dir, 
                                       "recons_mnist", 
                                       f"input_Epoch_{i}.png"),
                          normalize=True,
                          nrow=12)
                    x[random_indices_img_z] = img_gen
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

           
        print(('Epoch %d, loss = %.4f' % (i, loss.item())))
        check_accuracy(loader_val, model)
        print()
                


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



# data = VAEDataset(**config["data_params"], pin_memory=config['trainer_params']['gpus'])
# data.setup()

# loader_train = data.train_dataloader()
# loader_val = data.val_dataloader()

NUM_TRAIN = 59000
train_batch_size= config['data_params']['train_batch_size']
val_batch_size = config['data_params']['val_batch_size']
data_path = config['data_params']['data_path']
exp_flag = config['model_params']['exp_flag']
patch_size = config['data_params']['patch_size']

print("exp flag: ",exp_flag)

train_transform = transform = T.Compose([
                T.Resize(patch_size),
                T.ToTensor(),
                T.Normalize((0.5071), (0.2675))
                
            ])
mnist_train = dset.MNIST(data_path, train=True, download=True,
                             transform=train_transform)
loader_train = DataLoader(mnist_train, batch_size=train_batch_size, num_workers=2,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

mnist_val = dset.MNIST(data_path, train=True, download=True,
                          transform=transform)
loader_val = DataLoader(mnist_val, batch_size=val_batch_size, num_workers=2, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))

mnist_test = dset.MNIST(data_path, train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(mnist_test, batch_size=val_batch_size, num_workers=2)

in_channels= config['model_params']['in_channels']
num_class = config['model_params']['n_class']
learning_rate = config['exp_params']['LR']
weight_decay = config['exp_params']['weight_decay']
epochs = config['trainer_params']['max_epochs']

log_dir = config['logging_params']['save_dir']


Path(f"{log_dir}/input_mnist").mkdir(exist_ok=True, parents=True)
Path(f"{log_dir}/recons_mnist").mkdir(exist_ok=True, parents=True)

model = ClassifierMNIST(in_channels=in_channels,num_outputs=num_class)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_classifier(model,optimizer,epochs,exp_flag,latent_dim,vaemodel,acmodel)

#test
print("\n")
print("Test accuracy")
check_accuracy(loader_test, model)
                