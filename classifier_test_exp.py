import os
import math
import torch
from torch import optim
from models import BaseVAE, Actor
from models import *
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class ClassifierExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 vanilla_vae_model: BaseVAE,
                 actor: Actor,
                 exp_flag: int,
                 params: dict) -> None:
        super(ClassifierExperiment, self).__init__()

        self.model = vae_model
        self.vae_model = vanilla_vae_model
        self.actor = actor
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.exp_flag = exp_flag
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        
        real_img, labels = batch
        self.curr_device = real_img.device
        bs = real_img.shape[0]
        
        if self.exp_flag!=0:
            if self.exp_flag == 1:
                with torch.no_grad():
                    mu, var = self.vae_model.encode(real_img)
                    real_z = self.vae_model.reparameterize(mu, var)
                    noise = torch.randn(real_z.shape)
                    noise = noise.to(self.curr_device)
                    z = real_z+noise
                    img_gen = self.vae_model.decode(z)
            elif self.exp_flag == 2:  
                with torch.no_grad():
                    fake_z = torch.randn(bs, self.d_model)
                    fake_z = fake_z.to(self.device)
                    z_g = self.actor(fake_z, labels)
                    img_gen = self.vae_model.decode(z_g)
                
            random_indices_real_img = torch.randint(low=0, high=bs,size=(bs//2,))
            random_indices_real_img = random_indices_real_img.to(self.curr_device)
            random_indices_img_z = torch.randint(low=0, high=bs,size=(bs//2,))
            random_indices_img_z = random_indices_img_z.to(self.curr_device)
            
            real_img = torch.index_select(real_img, 0, random_indices_real_img)
            img_gen = torch.index_select(img_gen, 0, random_indices_img_z)
            real_img = torch.cat((real_img,img_gen), 0)
            
            labels_real = torch.index_select(labels, 0, random_indices_real_img)
            labels_img = torch.index_select(labels, 0, random_indices_img_z)
            labels = torch.cat((labels_real,labels_img), 0)
            
            real_img = real_img.to(self.curr_device)
            labels = labels.to(self.curr_device)
        
        
        results = self.forward(real_img)
        train_loss = self.model.loss_function(results,labels, #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(results,labels,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
