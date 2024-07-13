'''
Default config for DECA
 python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import pickle
import os
import torch

# torch.cuda.set_per_process_memory_fraction(0.85)  # Use up to 90% of GPU memory
# torch.cuda.set_per_process_memory_fraction(0.9, device=0)

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda:1'
cfg.device_id = '1,0'

#cfg.pretrained_modelpath = '../../data/deca_model.tar'
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/EM_AU_TENC_DEC_dimp_chin_EMOCA_encoder_coefs_all1_feature_consistency_loss_weighted_class_loss/model.tar'
# cfg.pretrained_modelpath = '/mnt/sdb/DJ_Training/AU_Tenc_dec_detailparam2313213536/model.tar'
cfg.pretrained_modelpath = '/mnt/sdb/DJ_Training/AU_Tenc_dec_detailparam2313213536_aufloss_300/model.tar'
# cfg.pretrained_modelpath = '/mnt/sdb/DJ_Training/WGAN_GP_with_AUCLOSS/model.tar'
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/DECA_origT/model.tar'
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/EM_AU_Step_ler_hptune_v1/model.tar' #4days model
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/AU_Loss_MEFL_MEFARG_feature_loss_and_dispparam_/model.tar' #ver_crossentropy_auFeature
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/wrinkle_enhanced_model/model.tar' #ver_without_AU_loss
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/train_only_au_loss/model.tar' #ver_2
# cfg.pretrained_modelpath = '/mnt/hdd/DJ_Training/AU_Loss_MEFARG_feature_loss_and_dispparam/model.tar' #ver_3
# cfg.pretrained_modelpath='/home/cine/DJ/DECA/Training/DECA_DJ2_additional_AU_Loss_plz/model.tar'
# cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
cfg.output_dir = ''
cfg.rasterizer_type = 'pytorch3d'
# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.deca_dir, 'data', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl') 
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy') 
cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png') 
cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png') 
cfg.model.mean_tex_path = os.path.join(cfg.deca_dir, 'data', 'mean_texture.jpg') 
cfg.model.tex_path = os.path.join(cfg.deca_dir, 'data', 'FLAME_albedo_from_BFM.npz') 
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.use_tex = True
cfg.model.extract_tex = True
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition model
cfg.model.fr_model_path = os.path.join(cfg.deca_dir, 'data', 'resnet50_ft_weight.pkl')

## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
# cfg.dataset.training_data = ['vggface2', 'ethnicity']
# cfg.dataset.training_data = ['ethnicity']
# cfg.dataset.eval_data = ['aflw2000']
# cfg.dataset.test_data = ['']
cfg.dataset.training_data = ['selfDataset']

cfg.dataset.mediapipePath = '/mnt/hdd/EncoderTrainingCode/Code/data/mediapipe_landmark_embedding.npz'
cfg.dataset.batch_size = 16
cfg.dataset.K = 1
cfg.dataset.isSingle = False
cfg.dataset.num_workers = 2
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.train_detail = True
cfg.train.max_epochs = 500
cfg.train.max_steps = 100000000
cfg.train.lr = 0.0001
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 200
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 500
cfg.train.val_steps = 500
cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 5000
cfg.train.resume = True

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.lmk = 1.0
cfg.loss.useWlmk = True
cfg.loss.eyed = 1.0
cfg.loss.lipd = 0.5
cfg.loss.photo = 2.0
cfg.loss.useSeg = True
cfg.loss.id = 0.2
cfg.loss.id_shape_only = True
cfg.loss.reg_shape = 1e-04
cfg.loss.reg_exp = 1e-04
cfg.loss.reg_tex = 1e-04
cfg.loss.reg_light = 1.
cfg.loss.reg_jaw_pose = 0. #1.
cfg.loss.use_gender_prior = False
cfg.loss.shape_consistency = True
# loss for detail
cfg.loss.detail_consistency = True
cfg.loss.useConstraint = True
cfg.loss.mrf = 5e-2
cfg.loss.photo_D = 2.0
cfg.loss.reg_sym = 0.005
cfg.loss.reg_z = 0.005
cfg.loss.reg_diff = 5e-3
cfg.loss.AU_feature= 0.25

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args(cfg_name=None):
    parser = argparse.ArgumentParser()
    if cfg_name == None:
        parser.add_argument('--cfg', type=str, default='configs/release_version/deca_detail.yml', help='cfg file path')
    else:
        parser.add_argument('--cfg', type=str, default=cfg_name, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='deca mode')


    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
