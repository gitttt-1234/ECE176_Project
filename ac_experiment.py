import os
import math
import copy
import torch
import numpy as np
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class ACEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 ac_model: BaseVAE,
                 params: dict) -> None:
        super(ACEXperiment, self).__init__()

        self.vae_model = vae_model
        self.ac_model = ac_model
        self.params = params
        self.d_model = self.params['latent_dim']
        self.curr_device = None
        self.hold_graph = False
        self._set_label_type()
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def _set_label_type(self):
        self.labels = torch.eye(self.params['num_labels'])
        self.labels = self.labels.to(self.device)
        self.num_labels = self.labels.size(0) 

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.vae_model(input, **kwargs)

    def fake_attr_generate(self, bs):
        selection = np.random.randint(self.num_labels, size=bs)
        selection = torch.from_numpy(selection)
        fake_attr = torch.index_select(self.labels, 0, selection)
        return fake_attr
    
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        bs = real_img.shape[0]
        real_data = torch.ones(bs,1)
        fake_data = torch.zeros(bs,1)
        fake_z = torch.randn(bs, self.d_model)
        fake_attr = self.fake_attr_generate(bs)
        with torch.no_grad():
            mu, var = self.vae_model.encode(real_img)
        z = self.vae_model.reparameterize(mu, var)
        
        self.curr_device = real_img.device

        if  np.random.rand(1) < 0.1:
            input_data = torch.cat([z, fake_z, z],dim=0) 
            input_attr = torch.cat([labels, labels, fake_attr],dim=0)
            real_labels = torch.cat([real_data, fake_data, fake_data])
        else:
            z_g = self.ac_model.actor_forward(fake_z, labels)		
            input_data = torch.cat([z, z_g, z],dim=0) 
            input_attr = torch.cat([labels,labels, fake_attr],dim=0)
            real_labels = torch.cat([real_data, fake_data, fake_data])
        
        logit_out = self.ac_model.real_critic_forward(input_data, input_attr)
        critic_loss = self.ac_model.critic_loss_function(logit_out, real_labels)
        loss = critic_loss['loss']
        
        if batch_idx > 0 and batch_idx % 10 == 0:
            fake_z = torch.randn(bs, self.d_model)
            fake_z = copy.deepcopy(fake_z)
            actor_z = self.ac_model.actor_forward(fake_z, labels) 
            real_z = self.ac_model.actor_forward(z, labels)
            zg_critic_out = self.ac_model.real_critic_forward(actor_z, labels)
            zg_critic_real = self.ac_model.real_critic_forward(real_z, labels)
            actor_loss = self.ac_model.actor_loss_function(var, z, fake_z, real_z, actor_z, zg_critic_out, zg_critic_real, real_data)
            loss += actor_loss['loss']
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        val_loss = self.vae_model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        bs = test_input.shape[0]
#         test_input, test_label = batch
        
        fake_z = torch.randn(bs, self.d_model)
        fake_z = fake_z.to(self.device)
        z_g = self.ac_model.actor_forward(fake_z, test_label)
        z_g_recon = self.vae_model.decode(z_g)
        prior_recon = self.vae_model.decode(fake_z)
        z, var = self.vae_model.encode(test_input)
        recons = self.vae_model.decode(z)
        vutils.save_image(z_g_recon.data,
                          os.path.join(self.logger.log_dir , 
                                       "Sample_Z_G", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        vutils.save_image(prior_recon.data,
                          os.path.join(self.logger.log_dir , 
                                       "Sample_Prior_G", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            for i in range(self.num_labels):
                test_labels = self.labels[i]
                test_labels = test_labels.expand(64,-1)

                sample = torch.randn(64, self.d_model).to(self.device)
                sample = self.ac_model.actor_forward(sample, test_labels)

                samples = self.vae_model.decode(sample)
                vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}_Testlabel_{i}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.ac_model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.ac_model,self.params['submodel']).parameters(),
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
