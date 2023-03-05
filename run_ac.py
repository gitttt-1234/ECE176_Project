import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from models import *
from ac import LitAC
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
                    default='configs/vae.yaml')
parser.add_argument('--vae-config',  '-vc',
                    dest="vfilename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

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


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

actor = vae_models[config['model_params']['actor_name']](**config['model_params'])
critic = vae_models[config['model_params']['critic_name']](**config['model_params'])
vaemodel = vae_models[vconfig['model_params']['name']](**vconfig['model_params'])
state_dict = torch.load(config['model_params']['vae_ckpt'])['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace("model.","")] = state_dict.pop(key)
vaemodel.load_state_dict(state_dict, strict=False)
experiment = LitAC(vaemodel,
                    actor,
                    critic,
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


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Sample_Z_G").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Sample_Prior_G").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)