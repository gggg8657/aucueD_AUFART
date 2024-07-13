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
# from gdl_apps.EMOCA.utils.load import load_model
# from gdl.datasets.ImageTestDataset import TestData
# import gdl
# import numpy as np
# import os
# import torch
# from skimage.io import imsave
# from pathlib import Path
# from tqdm import auto
# import argparse
# from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
# import os, sys
# import torch
# import torchvision
# import torch.nn.functional as F
# import torch.nn as nn

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from decalib.deca import DECA as DECA_
# from decalib.deca_AU_ import DECA as DECA_
from decalib.deca import DECA as DECA_
from decalib.deca_orig import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.config_orig import cfg as deca_cfg_orig
from decalib.utils.tensor_cropper import transform_points


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

from decalib.models.OpenGraphAU.model.MEFL import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.utils import *
from decalib.models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env
def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    ph_score_list =[]
    phdeca_score_list =[]

    auconf = get_config()
    auconf.evaluate = True
    auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    set_env(auconf)
    AU_net = MEFARG(num_main_classes=auconf.num_main_classes, num_sub_classes=auconf.num_sub_classes, backbone=auconf.arc).to(device)
    AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    AU_net.eval()
    
    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = "pytorch3d"
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA_(config = deca_cfg, device=device)
    deca_orig = DECA(config = deca_cfg_orig, device=device)
    emoca, conf = load_model(path_to_models='/mnt/hdd/emoca/assets/EMOCA/models/', run_name='EMOCA_v2_lr_mse_20', stage='detail')
    emoca.cuda()
    emoca.eval()

    # 2) Create a dataset
    # dataset = TestData(args.inputpath, face_detector="fan", max_detection=20)
    
    pathfile = open(args.inputpath, 'r')
    lines = pathfile.readlines()
    for line in lines:
        # print(line)
        pathname = line.split(',')[0]
        inputpath = line.split(',')[1].split('\n')[0]
        os.makedirs(os.path.join(savefolder, pathname), exist_ok=True)
        # os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)

        # os.makedirs(os.path.join(savefolder, 'obj'), exist_ok=True)
        # os.makedirs(os.path.join(savefolder, 'npy'), exist_ok=True)
        
        # load test images
        testdata = datasets.TestData(inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
        dataset = TestData(inputpath, face_detector="fan", max_detection=20)
        time_per_frame = 0.0
        b = 0.95
        # for i in range(len(testdata)):
        # for i in tqdm(range(len(testdata))):
        csvfile = open(os.path.join(savefolder,f'au_{pathname}.csv'), 'w')
        # csvfile.write('gt0,gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11,gt12,gt13, gt14, pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7,pd8,pd9,pd10,pd11,pd12,pd13,pd14,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dc8,dc9,dc10,dc11,dc12,dc13,dc14,emc0,emc1,emc2,emc3,emc4,emc5,emc6,emc7,emc8,emc9,emc10,emc11,emc12,emc13,emc14\n')
        csvfile.write('gt0,gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11,gt12,gt13, gt14, pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7,pd8,pd9,pd10,pd11,pd12,pd13,pd14,dc0,dc1,dc2,dc3,dc4,dc5,dc6,dc7,dc8,dc9,dc10,dc11,dc12,dc13,dc14\n')
        for i in tqdm(range(0, len(testdata), 5)):
            batch = dataset[i]
            name = testdata[i]['imagename']
            images = testdata[i]['image'].to(device)[None,...]
            cond=(AU_net(images)[1]>=0.7).int().float()
            if torch.all(cond==0):
                print(cond)
                print("\n\n\n\nSKIP!!!!!\n\n\n")
                continue
            with torch.no_grad():
                try: 
                    codedict= deca.encode(images, batch=batch)
                    print("try success")
                except Exception as e : 
                    print(f"try failed{e}")
                    codedict = deca.encode(images)
                
                # print(codedict)
                visdict = {}
                visdict['inputs'] = images
                visdict2 = {}
                visdict2['inputs'] = images
                # _, visdict_emo = test(emoca, batch)
                # codedict_emo, _ = emoca.encode(batch=batch, training=False)
                # codedict_emo['shape'] = codedict_emo['shapecode']
                # codedict_emo['tex'] = codedict_emo['texcode']
                # codedict_emo['pose'] = codedict_emo['posecode']
                # codedict_emo['exp'] = codedict_emo['expcode']
                # codedict_emo['cam'] = codedict_emo['cam']
                # codedict_emo['light'] = codedict_emo['lightcode']
                # codedict_emo['detail'] = codedict_emo['detailcode']

                codedict_orig = deca_orig.encode(images)
                opdict_orig, visdict_orig, visdict2['deca'], ph_d = deca_orig.decode(codedict_orig) #tensor
                phdeca_score_list.append(ph_d)
                # opdict_emo, visdict_emo, visdict2['emoca'] = deca_orig.decode(codedict_emo)

                # codedict_orig = deca_orig.encode(images)
                # opdict_orig, visdict_orig, visdict2['deca'] = deca_orig.decode(codedict_orig) #tensor

                visdict['Deca_detail'] = visdict_orig['final_images']
                # visdict['EMOCA_detail'] = visdict_emo['final_images']
                opdict, visdict['shape_detail_images'], visdict2['My'], ph_score = deca.decode(codedict) #tensor
                ph_score_list.append(ph_score)
                
                
                # print(visdict['shape_detail_images'].shape)
                # print(visdict['shape_detail_images'])
                if args.render_orig:
                    tform = testdata[i]['tform'][None, ...]
                    tform = torch.inverse(tform).transpose(1,2).to(device)
                    original_image = testdata[i]['original_image'][None, ...].to(device)
                    _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
                    _, orig_visdict_orig = deca_orig.decode(codedict_orig, render_orig=True, original_image=original_image, tform=tform)    
                    
                    

                    orig_visdict['inputs'] = original_image
                    orig_visdict['Deca_detail'] = orig_visdict_orig['shape_detail_images']
                    # orig_visdict['EMOCA_detail'] = visdict_emo['output_images_detail']
                    orig_visdict_orig['inputs'] = original_image            
                compare = [0, 1, 2, 3, 5, 6, 7, 9,11, 12, 13,14, 17, 19, 22]
                image_au = AU_net(visdict['inputs'])[1][:, compare]
                rend_au = (AU_net(visdict['shape_detail_images'])[1][:, compare]>=0.5).int().float()
                rend_au_deca = (AU_net(visdict['Deca_detail'])[1][:, compare]>=0.5).int().float()
                # rend_au_emoca = (AU_net(visdict['EMOCA_detail'])[1][:, compare]>=0.5).int().float()

                for j in range(len(compare) * 4):  # Adjusted to *4 for all conditions
                    if j < len(compare):
                        csvfile.write(f'{image_au[0, j].item()},')
                        # print(f'Image AU: {image_au[0, j].item()}')
                    elif j < len(compare) * 2:
                        csvfile.write(f'{rend_au[0, j - len(compare)].item()},')
                        # print(f'Rend AU: {rend_au[0, j - len(compare)].item()}')
                    elif j < len(compare) * 3:
                        csvfile.write(f'{rend_au_deca[0, j - len(compare) * 2].item()},')
                        # print(f'Rend AU DECA: {rend_au_deca[0, j - len(compare) * 2].item()}')
                    # else:
                    #     csvfile.write(f'{rend_au_emoca[0, j - len(compare) * 3].item()},')
                        # print(f'Rend AU EMOCA: {rend_au_emoca[0, j - len(compare) * 3].item()}')
                csvfile.write('\n')
            # if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            #     os.makedirs(os.path.join(savefolder, name), exist_ok=True)
            # -- save results
            # if args.saveDepth:
                
            #     depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            #     visdict['depth_images'] = depth_image
            #     cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
            #     depth_image_orig = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            #     visdict_orig['depth_images'] = depth_image_orig
            #     cv2.imwrite(os.path.join(savefolder, name, name + '_depth_deca.jpg'), util.tensor2image(depth_image[0]))
            # if args.saveKpt:
            #     np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            #     np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
            # if args.saveObj:
            #     deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
            #     deca_orig.save_obj(os.path.join(savefolder, name, name + '_deca.obj'), opdict)
            # if args.saveMat:
            #     opdict = util.dict_tensor2npy(opdict)
            #     savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
            
            if args.saveVis:
                shapefolder = os.path.join(savefolder, 'shape')
                os.makedirs(shapefolder, exist_ok=True)
                cv2.imwrite(os.path.join(shapefolder, name + '_shape_vis.jpg'), deca.visualize(visdict2))
                print(os.path.join(shapefolder,name))
                cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
                if args.render_orig:
                    cv2.imwrite(os.path.join(savefolder, name + '_shape_vis_original_size.jpg'), deca.visualize(visdict2))
                    cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(visdict))
            if args.saveImages:
                for vis_name in ['inputs', 'Deca_detail', 'EMOCA_detail','shape_detail_images']:
                    if vis_name not in visdict.keys():
                        continue
                    # image = util.tensor2image(visdict[vis_name][0])
                    # tmp = os.path.join(savefolder, vis_name, '/')
                    # print(tmp)
                    # cv2.imwrite(os.path.join(tmp, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                    dir_path = os.path.join(savefolder, vis_name)
                    os.makedirs(dir_path, exist_ok=True)
                    
                    # 이미지 저장 경로 생성
                    image_path = os.path.join(dir_path, name + '_' + vis_name + '.jpg')
                    image = util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(image_path, image)
                    # if args.render_orig:
                    #     image = util.tensor2image(orig_visdict[vis_name][0])
                    #     cv2.imwrite(os.path.join(savefolder, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
        
        csvfile.close()
        print(f'-- please check the results in {savefolder}')
    ph_min = min(ph_score_list)
    ph_max = max(ph_score_list)
    average = sum(ph_score_list) / len(ph_score_list)
    ph_score_list.append(average)
    phdeca_score_list.append(ph_min)
    phdeca_score_list.append(ph_max)
    ph_min2 = min(phdeca_score_list)
    ph_max2 = max(phdeca_score_list)
    average2 = sum(phdeca_score_list) / len(phdeca_score_list)
    phdeca_score_list.append(ph_min2)
    phdeca_score_list.append(ph_max2)
    phdeca_score_list.append(average2)
    
    items = ph_score_list
    file = open('my_model_ph_list.txt','w')
    for it in items:
        it = it.item()
        file.write(str(it)+"\n")
        # print(item.item()+"\n")
    file.close()
    csvfile.close()
    items2 = phdeca_score_list
    file = open('DECA_ph_list.txt','w')
    for it in items2:
        it = it.item()
        file.write(str(it)+"\n")
        # print(item.item()+"\n")
    file.close()
    pathfile.close()
        
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    # parser.add_argument('-i', '--inputpath', default='TestSamples/examples/chin_dim', type=str,
    parser.add_argument('-i', '--inputpath', default='/home/cine/DJ/DECA/demos/paths.txt', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/new_results_tmp', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())