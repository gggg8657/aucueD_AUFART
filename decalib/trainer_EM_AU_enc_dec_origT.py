# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .datasets import datasets
from .utils.config import cfg
torch.backends.cudnn.benchmark = True
from .utils import lossfunc_AU_enc_dec_origT as lossfunc

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test


# from .models.OpenGraphAU.model.ANFL import AFG
from .models.OpenGraphAU.model.MEFL import MEFARG
from .models.OpenGraphAU.utils import load_state_dict
from .models.OpenGraphAU.utils import *
from .models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env

from .datasets import build_datasets_detail as build_datasets
# from .datasets import build_datasets

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
class Trainer(object):
    def __init__(self, model, config=None, device='cuda'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.K = self.cfg.dataset.K
        # training stage: coarse and detail
        self.train_detail = self.cfg.train.train_detail

        # deca model
        self.deca = model.to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        # self.emoca, conf = load_model(path_to_models='/mnt/hdd/emoca/assets/EMOCA/models/', run_name='EMOCA_v2_lr_mse_20', stage='detail')
        # self.emoca.cuda()
        # self.emoca.eval()
        # initialize loss  
        # # initialize loss   
        if self.train_detail:     
            self.mrf_loss = lossfunc.IDMRFLoss()
            # self.au_feature_loss=lossfunc.AU_Feature_Loss_()
            self.au_feature_loss=lossfunc.AU_Feature_Loss()
            # self.vggface2_loss = lossfunc.VGGFace2Loss(pretrained_model='/home/cine/DJ/DECA/data/resnet50_ft_weight.pkl')
            # self.per_loss =  lossfunc.PerceptualLoss() 
            self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        else:
            self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      
        
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
    
    def configure_optimizers(self):
        
        self.opt = torch.optim.AdamW(
                                list(self.deca.E_detail.parameters()) + \
                                list(self.deca.D_detail.parameters()),# + \
                                # list(self.deca.AUEncoder.parameters()),
                                # list(self.deca.AUNet.parameters()),
                                # list()
                                # list(self.deca.DAT.parameters()),
                                lr=self.cfg.train.lr,
                                amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10000, gamma=0.999)  
        # else:
        #     self.opt = torch.optim.Adam(
        #                             self.deca.E_flame.parameters(),
        #                             lr=self.cfg.train.lr,
        #                             amsgrad=False)
    def load_checkpoint(self):
        self.auconf = get_config()
        model_dict = self.deca.model_dict()
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0
        # self.AU_net = MEFARG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes, backbone=self.auconf.arc).to(self.device)
        # self.AU_net = load_state_dict(self.AU_net, self.auconf.resume).to(self.device)
        # self.AU_net.eval()

    def training_step(self, batch, batch_nb, training_type='coarse'):
        self.deca.train()
        
        if self.train_detail:
            self.deca.E_flame.eval()
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = batch['image_224'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
        lmk = batch['landmark'].to(self.device); lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        masks = batch['mask'].to(self.device); masks = masks.view(-1, images.shape[-2], images.shape[-1]) 

        #-- encoder
        # codedict = self.emoca.encode(batch=batch, training=False) #ver a
        codedict = self.deca.encode(images, batch,use_detail=self.train_detail)
        
        ### shape constraints for coarse model
        ### detail consistency for detail model
        # import ipdb; ipdb.set_trace()
        self.cfg.loss.shape_consistency =False 
        self.cfg.loss.detail_consistency = False
        if self.cfg.loss.shape_consistency or self.cfg.loss.detail_consistency:
            '''
            make sure s0, s1 is something to make shape close
            the difference from ||so - s1|| is 
            the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
            '''
            new_order = np.array([np.random.permutation(self.K) + i*self.K for i in range(self.batch_size)])
            new_order = new_order.flatten()
            shapecode = codedict['shape']
            if self.train_detail:
                detailcode = codedict['detailcode']
                detailcode_new = detailcode[new_order]
                codedict['detailcode'] = torch.cat([detailcode, detailcode_new], dim=0)
                codedict['shapecode'] = torch.cat([shapecode, shapecode], dim=0)
            else:
                print("traine coarse")
                # shapecode_new = shapecode[new_order]
                # codedict['shapecode'] = torch.cat([shapecode, shapecode_new], dim=0)
            # for key in ['texcode', 'expcode', 'posecode', 'cam', 'lightcode', 'images']:
            for key in ['tex', 'exp', 'pose', 'cam', 'light', 'images']:
                code = codedict[key]
                codedict[key] = torch.cat([code, code], dim=0)
            ## append gt
            images = torch.cat([images, images], dim=0)# images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
            lmk = torch.cat([lmk, lmk], dim=0) #lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
            masks = torch.cat([masks, masks], dim=0)

        batch_size = images.shape[0]

        ###--------------- training coarse model
        if not self.train_detail:
            # #-- decoder
            print("train coarse")
           
        ###--------------- training detail model
        else:
            #-- decoder
            shapecode = codedict['shape']
            # expcode = codedict['exp'] #current -> expression code used as input of detail decoders
            posecode = codedict['pose']
            texcode = codedict['tex']
            lightcode = codedict['light']
            detailcode = codedict['detailcode']
            cam = codedict['cam']
            # shapecode = codedict['shapecode']
            expcode = codedict['exp']
            # posecode = codedict['posecode']
            # texcode = codedict['texcode']
            # lightcode = codedict['lightcode']
            # detailcode = codedict['detailcode']
            # cam = codedict['cam']


            # FLAME - world space
            verts, landmarks2d, landmarks3d= self.deca.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
            landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:] #; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
            # mp_landmark = util.batch_orth_proj(mp_landmark, codedict['cam'])[:,:,:2]; mp_landmark[:,:,1:] = -mp_landmark[:,:,1:]
            # world to camera
            trans_verts = util.batch_orth_proj(verts, cam)
            predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]
            # camera to image space
            trans_verts[:,:,1:] = -trans_verts[:,:,1:]
            predicted_landmarks[:,:,1:] = - predicted_landmarks[:,:,1:]
            
            albedo = self.deca.flametex(texcode)

            #------ rendering
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), ops['grid'].detach(), align_corners=False)
            # images
            predicted_images = ops['images']*mask_face_eye*ops['alpha_images']

            masks = masks[:,None,:,:]
            codedict['AU_features'], codedict['afn'], _= self.deca.AUNet(images)
            
            # afn >= 0.3 go to 1 

            # codedict['encoded_AUF'] = self.deca.AUEncoder(codedict['AU_features'])
            # codedict['detailcode'] = self.deca.DAT(codedict['au_feature'], codedict['detail_features'])
            # uv_z = self.deca.D_detail(torch.cat([posecode[:,3:], expcode, codedict['detailcode'],codedict['afn'],codedict['encoded_AUF']], dim=1))
            uv_z = self.deca.D_detail(torch.cat([posecode[:,3:].unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,1), expcode.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,1), codedict['detailcode']], dim=1), codedict['afn'].unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,1))
            # render detail
            uv_detail_normals = self.deca.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
            uv_texture = albedo.detach()*uv_shading
            predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)
            detail_normal_image = F.grid_sample(uv_detail_normals, ops['grid'].detach(), align_corners=False)
            shape_detail_images = self.deca.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_image, images=images)
            
            # shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w, images=images, return_grid=True)
            # detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)*alpha_images
            # shape_detail_images = self.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images, h=h, w=w, images=background)
            # shape_detail_images_albedo = self.deca.render.render_shape(verts, trans_verts, detail_normal_images=predicted_detail_images, images=images)
            
            #--- extract texture
            uv_pverts = self.deca.render.world2uv(trans_verts).detach()
            uv_gt = F.grid_sample(torch.cat([images, masks], dim=1), uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)
            uv_texture_gt = uv_gt[:,:3,:,:].detach(); uv_mask_gt = uv_gt[:,3:,:,:].detach()
            # self-occlusion
            normals = util.vertex_normals(trans_verts, self.deca.render.faces.expand(batch_size, -1, -1))
            uv_pnorm = self.deca.render.world2uv(normals)
            uv_mask = (uv_pnorm[:,[-1],:,:] < -0.05).float().detach()
            ## combine masks
            uv_vis_mask = uv_mask_gt*uv_mask*self.deca.uv_face_eye_mask
            
            #### ----------------------- Losses
            losses = {}
            ############################### details
            # if self.cfg.loss.old_mrf: 
            #     if self.cfg.loss.old_mrf_face_mask:
            #         masks = masks*mask_face_eye*ops['alpha_images']
            #     losses['photo_detail'] = (masks*(predicted_detailed_image - images).abs()).mean()*100
            #     losses['photo_detail_mrf'] = self.mrf_loss(masks*predicted_detailed_image, masks*images)*0.1
            # else:
            pi = 0
            new_size = 256
            
            uv_texture_patch = F.interpolate(uv_texture[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            uv_texture_gt_patch = F.interpolate(uv_texture_gt[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            uv_vis_mask_patch = F.interpolate(uv_vis_mask[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            #
            # images_tmp = F.interpolate(images, size = [new_size, new_size], mode='bilinear', align_corners=False)
            # predicted_detail_images_tmp = F.interpolate(predicted_detail_images, size = [new_size, new_size], mode='bilinear', align_corners=False)
            final_image = predicted_detail_images * masks + images*(1- masks) #go to _adv and make discriminator
            # uv_tmp = F.interpolate(uv_vis_mask, size=(224,224), mode='bilinear', align_corners=False)
            # final_image = predicted_detail_images * uv_tmp + images*(1 - uv_tmp)
            # with autocast():
            losses['photo_detail'] = ((uv_texture_patch*uv_vis_mask_patch)**2 - (uv_texture_gt_patch*uv_vis_mask_patch)**2).abs().mean()*self.cfg.loss.photo_D*10
            

        

            #old mrf
            # masks = masks*mask_face_eye*ops['alpha_images']
            # losses['photo_detail_mrf'] = self.mrf_loss(uv_texture_patch*uv_vis_mask_patch, masks*uv_texture_gt_patch*uv_vis_mask_patch)*0.1
            # losses['perceptual_detail'] = self.per_loss(predicted_detail_images, images)
            losses['photo_detail_mrf'] = self.mrf_loss(uv_texture_patch*uv_vis_mask_patch, uv_texture_gt_patch*uv_vis_mask_patch)*self.cfg.loss.photo_D*self.cfg.loss.mrf
            # losses['au_feature_loss'] = self.au_feature_loss(uv_texture_patch*uv_vis_mask_patch, uv_texture_gt_patch*uv_vis_mask_patch) # ver1
            # losses['au_feature_loss'], losses['chin_loss'], losses['dimp_loss'],losses['au_class_loss']= self.au_feature_loss(self.deca.AUNet(images), self.deca.AUNet(predicted_detail_images)) # ver
            #  2
            # losses['au_class_loss']= self.au_feature_loss(self.deca.AUNet(images), self.deca.AUNet(predicted_detail_images)) 
            # losses['au_class_consistency_loss']= self.au_feature_loss(self.deca.AUNet(images)[1], self.deca.AUNet(predicted_detail_images)[1]) #ver 3
            # losses['au_class_consistency_loss'], losses['chin_loss'], losses['dimp_loss']= self.au_feature_loss(self.deca.AUNet(images)[1], self.deca.AUNet(predicted_detail_images)[1]) #ver 3
            # losses['vggface2_detail'] = self.vggface2_loss(predicted_detail_images, images)*self.cfg.loss.photo_D
            
            losses['z_reg'] = torch.mean(uv_z.abs())*self.cfg.loss.reg_z
            losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading)*self.cfg.loss.reg_diff
            if self.cfg.loss.reg_sym > 0.:
                nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
                losses['z_sym'] = (nonvis_mask*(uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum()*self.cfg.loss.reg_sym
        # img_afn = self.AU_net(images, use_gnn=True)[2]
        # # img_afn = torch.round(img_afn*10, decimals=0)
        # rend_afn = self.AU_net(predicted_detail_images, use_gnn=True)[2]
        # # rend_afn = torch.round(rend_afn*10, decimals=0)
        # losses['AU_feature'] = F.mse_loss(img_afn,rend_afn)*50
        #original opdict location
            opdict = {
                'verts': verts,
                'trans_verts': trans_verts,
                'landmarks2d': landmarks2d,
                # 'mp_landmark': mp_landmark,
                'predicted_deatil_shape' : shape_detail_images ,
                'predicted_images': predicted_images,
                'predicted_detail_images': final_image,
                'images': images,
                'lmk': lmk
            }
            
        #########################################################
            all_loss = 0.
            losses_key = losses.keys()
            for key in losses_key:
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            return losses, opdict
        

    def prepare_data(self):
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        # self.val_dataset = build_datasets.build_val(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        # self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
        #                     num_workers=8,
        #                     pin_memory=True,
        #                     drop_last=False)
        # self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch, step)
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)                    
                    logger.info(loss_info)

                if self.global_step % self.cfg.train.vis_steps == 0:
                    print(f'Epoch: {epoch}, LR: {self.scheduler.get_last_lr()}')
                    visind = list(range(4))
                    shape_images = self.deca.render.render_shape(opdict['verts'][visind], opdict['trans_verts'][visind])
                    visdict = {
                        'inputs': opdict['images'][visind], 
                        # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                        # 'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                        'shape_images': shape_images, 
                    }
                    # if 'predicted_images' in opdict.keys():
                        
                        # visdict['predicted_images'] = opdict['predicted_images'][visind]
                    if 'predicted_detail_images' in opdict.keys():
                        visdict['predicted_detail_shape'] = opdict['predicted_deatil_shape'][visind]
                        visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]

                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                    grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
                    # import ipdb; ipdb.set_trace()                    
                    self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.deca.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))   
                    # 
                    if self.global_step % self.cfg.train.checkpoint_steps*10 == 0:
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))   

                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step()
                
                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                all_loss = losses['all_loss']

                # Scales the loss, and calls backward() on the scaled loss to create scaled gradients.
                scaler.scale(all_loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(self.opt)

                # Step with the optimizer
                scaler.step(self.opt)
                

                # Updates the scale for next iteration
                scaler.update()

                self.opt.zero_grad()
                self.scheduler.step()
                self.global_step += 1
                # self.opt.zero_grad(); all_loss.backward(); self.opt.step()
                # self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break
            