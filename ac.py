import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable, grad
from models import BaseVAE, Actor, Critic
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class LitAC(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 actor: Actor,
                 critic: Critic,
                 params: dict) -> None:
        super(LitAC, self).__init__()

        self.vae_model = vae_model
        self.actor = actor
        self.critic = critic
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
    
    def critic_loss_function(self,
                      pred,
                      target,
                      all_z,
                      all_attr,) -> dict:
        """
        :return:
        """
        loss = F.binary_cross_entropy(pred, target, size_average=True)
        bs = all_z.shape[0]
        real_data = all_z[:bs // 3]
        fake_data = all_z[bs // 3: (2 * bs) // 3]
        real_attr = all_attr[:bs // 3]
        alpha = torch.rand((real_data.shape[0], self.d_model))
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        interpolates = Variable(interpolates, requires_grad=True)
        interp_pred = self.critic(interpolates, real_attr)
        slopes = (grad(interp_pred.sum(), interpolates)[0].square().sum(dim=-1) + 1e-10).sqrt()
        gradient_penalty = torch.mean((slopes - 1.)**2)
        return {'loss': loss + 10 * gradient_penalty, 
                    'critic_loss': loss,
                    'gradient_loss': gradient_penalty}
    
    def actor_loss_function(self, var,
                                z, 
                                real_z, 
                                fake_z,
                                z_critic_out, 
                                z_critic_real, 
                                actor,):
        weight_var = torch.mean(var, 0, True)
        distance_penalty = torch.mean(torch.log(1 + (z - fake_z).pow(2)) * weight_var.pow(-2))
        distance_penalty += torch.mean(torch.log(1 + (real_z - z).pow(2)) * weight_var.pow(-2))
        
        # actor_loss = -torch.mean(torch.clip(F.sigmoid(z_critic_out), 1e-15, 1 - 1e-15).log()) \
        #                 -torch.mean(torch.clip(F.sigmoid(z_critic_real), 1e-15, 1 - 1e-15).log())
        actor_loss = F.binary_cross_entropy(z_critic_out, actor, size_average=False)\
                        + F.binary_cross_entropy(z_critic_real, actor, size_average=False)
        # print(actor_loss.shape, distance_penalty.shape)
        return {'distance_penalty': distance_penalty, 
                    'actor_loss': actor_loss, 
                    'loss': actor_loss + 0.00001 * distance_penalty}
        
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, real_attr = batch
        bs = real_img.shape[0]
        real_r = torch.ones(bs,1)
        fake_r = torch.zeros(bs,1)
        fake_z_prior = torch.randn(bs, self.d_model)
        fake_attr = self.fake_attr_generate(bs)
        with torch.no_grad():
            mu, var = self.vae_model.encode(real_img)
        real_z = self.vae_model.reparameterize(mu, var)
        
        self.curr_device = real_img.device

        if  np.random.rand(1) < 0.1:
            all_z = torch.cat([real_z, fake_z_prior, real_z], dim=0) 
        else:
            z_g = self.actor(fake_z_prior, real_attr)		
            all_z = torch.cat([real_z, z_g, real_z], dim=0) 
        all_attr = torch.cat([real_attr, real_attr, fake_attr],dim=0)
        all_r = torch.cat([real_r, fake_r, fake_r])
        
        logit_out = self.critic(all_z, all_attr)
        critic_loss = self.critic_loss_function(logit_out, 
                                                    all_r,
                                                    all_z,
                                                    all_attr)
        loss = critic_loss['loss']
        
        if batch_idx > 0 and batch_idx % 10 == 0:
            # fake_z = torch.randn(bs, self.d_model)
            # fake_z = copy.deepcopy(fake_z)
            fake_z_gen = self.actor(fake_z_prior, real_attr) 
            real_z_gen = self.actor(real_z, real_attr)
            zg_critic_out = self.critic(fake_z_gen, real_attr)
            zg_critic_real = self.critic(real_z_gen, real_attr)
            actor_loss = self.actor_loss_function(var, 
                                                                real_z,
                                                                real_z_gen, 
                                                                fake_z_gen, 
                                                                zg_critic_out, 
                                                                zg_critic_real, 
                                                                real_r)
            loss += actor_loss['loss']
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, real_attr = batch
        bs = real_img.shape[0]
        fake_z_prior = torch.randn(bs, self.d_model)
        with torch.no_grad():
            mu, var = self.vae_model.encode(real_img)
        real_z = self.vae_model.reparameterize(mu, var)
        real_z_gen = self.actor(real_z, real_attr)
        fake_z_gen = self.actor(fake_z_prior, real_attr)
        pred_eval = self.critic(real_z, real_attr)
        pred_gen = self.critic(fake_z_gen, real_attr)
        self.curr_device = real_img.device

        self.log_dict({f"val_z": torch.mean(pred_eval).item()}, sync_dist=True)
        self.log_dict({f"val_loss": F.mse_loss(real_z, real_z_gen, reduction='mean').item()}, sync_dist=True)

        
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
        z_g = self.actor(fake_z, test_label)
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
                sample = self.actor(sample, test_labels)

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

        optimizer = optim.Adam(self.actor.parameters(),
                               lr=self.params['actor_LR'],
                               weight_decay=self.params['weight_decay'])
        optimizer = optim.Adam(self.critic.parameters(),
                               lr=self.params['critic_LR'],
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
