''' training script of DECA
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
torch.cuda.empty_cache()
import shutil
import torch.nn as nn
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    # creat folders 
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    #os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
    
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # start training
    # deca model
    #train with AU
    from decalib.deca_AU import DECA
    from decalib.trainer_AU import Trainer
    #train origin DECA
    # from decalib.deca import DECA
    # from decalib.trainer import Trainer
    
    cfg.rasterizer_type = 'pytorch3d'
    cfg.device = 'cuda'
    deca = DECA(cfg)
    trainer = Trainer(model=deca, config=cfg)

    ## start train
    trainer.fit()

if __name__ == '__main__':
    from decalib.utils.config import parse_args
    cfg = parse_args(cfg_name='configs/release_version/deca_detail_AU_Loss.yml')

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml 