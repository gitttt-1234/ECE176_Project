import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from models import *
from classifier_test_exp import ClassifierExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
# from pytorch_lightning.strategies import DDPStrategy


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/classifier.yaml')
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

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])



vaemodel = vae_models[vconfig['model_params']['name']](**vconfig['model_params'])
state_dict = torch.load(config['model_params']['vae_ckpt'])['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace("model.","")] = state_dict.pop(key)
vaemodel.load_state_dict(state_dict, strict=False)

acmodel = vae_models[acconfig['model_params']['actor_name']](**acconfig['model_params'])
state_dict = torch.load(config['model_params']['ac_ckpt'])['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace("model.","")] = state_dict.pop(key)
acmodel.load_state_dict(state_dict, strict=False)



#exp_flag = config['model_params']['exp_flag']
exp_flag=1
experiment = ClassifierExperiment(model,vaemodel,acmodel,exp_flag,
                    config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=config['trainer_params']['gpus'])

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True,
                                     mode='min'),
                 ],
                #  strategy=DDPStrategy(find_unused_parameters=False),
                 **config['trainer_params'])



print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)