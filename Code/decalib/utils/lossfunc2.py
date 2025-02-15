import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from functools import reduce
import torchvision.models as models
import cv2
import torchfile
from torch.autograd import Variable

from focal_frequency_loss import FocalFrequencyLoss as FFL
from . import util
#[276 282 283 285 293 295 296 300 334 336  46  52  53  55  63  65  66  70
 # 105 107 249 263 362 373 374 380 381 382 384 385 386 387 388 390 398 466
 #   7  33 133 144 145 153 154 155 157 158 159 160 161 163 173 246 168   6
 # 197 195   5   4 129  98  97   2 326 327 358   0  13  14  17  37  39  40
 #  61  78  80  81  82  84  87  88  91  95 146 178 181 185 191 267 269 270
 # 291 308 310 311 312 314 317 318 321 324 375 402 405 409 415]

def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()

### VAE
def kl_loss(texcode):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mu, logvar = texcode[:,:128], texcode[:,128:]
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return KLD

### ------------------------------------- Losses/Regularizations for shading
# white shading
# uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)
# mean_shade = tf.reduce_mean( tf.multiply(shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
# G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
def shading_white_loss(shading):
    '''
    regularize lighting: assume lights close to white 
    '''
    # rgb_diff = (shading[:,0] - shading[:,1])**2 + (shading[:,0] - shading[:,2])**2 + (shading[:,1] - shading[:,2])**2
    # rgb_diff = (shading[:,0].mean([1,2]) - shading[:,1].mean([1,2]))**2 + (shading[:,0].mean([1,2]) - shading[:,2].mean([1,2]))**2 + (shading[:,1].mean([1,2]) - shading[:,2].mean([1,2]))**2
    # rgb_diff = (shading.mean([2, 3]) - torch.ones((shading.shape[0], 3)).float().cuda())**2
    rgb_diff = (shading.mean([0, 2, 3]) - 0.99)**2
    return rgb_diff.mean()

def shading_smooth_loss(shading):
    '''
    assume: shading should be smooth
    ref: Lifting AutoEncoders: Unsupervised Learning of a Fully-Disentangled 3D Morphable Model using Deep Non-Rigid Structure from Motion
    '''
    dx = shading[:,:,1:-1,1:] - shading[:,:,1:-1,:-1]
    dy = shading[:,:,1:,1:-1] - shading[:,:,:-1,1:-1]
    gradient_image = (dx**2).mean() + (dy**2).mean()
    return gradient_image.mean()

### ------------------------------------- Losses/Regularizations for albedo
# texture_300W_labels_chromaticity = (texture_300W_labels + 1.0)/2.0
# texture_300W_labels_chromaticity = tf.divide(texture_300W_labels_chromaticity, tf.reduce_sum(texture_300W_labels_chromaticity, axis=[-1], keep_dims=True) + 1e-6)


# w_u = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :-1, :, :] - texture_300W_labels_chromaticity[:, 1:, :, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :-1, :, :] )
# G_loss_local_albedo_const_u = tf.reduce_mean(norm_loss( albedo_300W[:, :-1, :, :], albedo_300W[:, 1:, :, :], loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_u) / tf.reduce_sum(w_u+1e-6)

    
# w_v = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :, :-1, :] - texture_300W_labels_chromaticity[:, :, 1:, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :, :-1, :] )
# G_loss_local_albedo_const_v = tf.reduce_mean(norm_loss( albedo_300W[:, :, :-1, :], albedo_300W[:, :, 1:, :],  loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_v) / tf.reduce_sum(w_v+1e-6)

# G_loss_local_albedo_const = (G_loss_local_albedo_const_u + G_loss_local_albedo_const_v)*10

def albedo_constancy_loss(albedo, alpha = 15, weight = 1.):
    '''
    for similarity of neighbors
    ref: Self-supervised Multi-level Face Model Learning for Monocular Reconstruction at over 250 Hz
        Towards High-fidelity Nonlinear 3D Face Morphable Model
    '''
    albedo_chromaticity = albedo/(torch.sum(albedo, dim=1, keepdim=True) + 1e-6)
    weight_x = torch.exp(-alpha*(albedo_chromaticity[:,:,1:,:] - albedo_chromaticity[:,:,:-1,:])**2).detach()
    weight_y = torch.exp(-alpha*(albedo_chromaticity[:,:,:,1:] - albedo_chromaticity[:,:,:,:-1])**2).detach()
    albedo_const_loss_x = ((albedo[:,:,1:,:] - albedo[:,:,:-1,:])**2)*weight_x
    albedo_const_loss_y = ((albedo[:,:,:,1:] - albedo[:,:,:,:-1])**2)*weight_y
    
    albedo_constancy_loss = albedo_const_loss_x.mean() + albedo_const_loss_y.mean()
    return albedo_constancy_loss*weight

def albedo_ring_loss(texcode, ring_elements, margin, weight=1.):
        """
            computes ring loss for ring_outputs before FLAME decoder
            Inputs:
              ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
              Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
              Aim is to force each row (same subject) of each stream to produce same shape
              Each row of first N-1 strams are of the same subject and
              the Nth stream is the different subject
        """
        tot_ring_loss = (texcode[0]-texcode[0]).sum()
        diff_stream = texcode[-1]
        count = 0.0
        for i in range(ring_elements - 1):
            for j in range(ring_elements - 1):
                pd = (texcode[i] - texcode[j]).pow(2).sum(1)
                nd = (texcode[i] - diff_stream).pow(2).sum(1)
                tot_ring_loss = torch.add(tot_ring_loss,
                                (torch.nn.functional.relu(margin + pd - nd).mean()))
                count += 1.0

        tot_ring_loss = (1.0/count) * tot_ring_loss
        return tot_ring_loss * weight

def albedo_same_loss(albedo, ring_elements, weight=1.):
        """
            computes ring loss for ring_outputs before FLAME decoder
            Inputs:
              ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
              Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
              Aim is to force each row (same subject) of each stream to produce same shape
              Each row of first N-1 strams are of the same subject and
              the Nth stream is the different subject
        """
        loss = 0
        for i in range(ring_elements - 1):
            for j in range(ring_elements - 1):
                pd = (albedo[i] - albedo[j]).pow(2).mean()
                loss += pd
        loss = loss/ring_elements
        return loss * weight

### ------------------------------------- Losses/Regularizations for vertices
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None and len(weights.shape) < 2:
        real_2d_kp[:,:,2] = weights[None,:]*real_2d_kp[:,:,2]
    elif weights is not None and len(weights.shape) >= 2:
        real_2d_kp[:,:,2] = weights[:,:]*real_2d_kp[:,:,2]

    kp_gt = real_2d_kp.view(-1, 3)
    # print('true_content1_1_1', real_2d_kp[0][1][1])
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    # print('kpt_content1_1_1', kp_pred[0][1][1])
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8

    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # print('real_2d_1',landmarks_gt.shape)
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    # print('real_2d_2',real_2d.shape)
    real_2d = landmarks_gt
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight

def landmark_HRNet_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # print('real_2d_1',landmarks_gt.shape)
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    # print('real_2d_2',real_2d.shape)
    weights = torch.ones((68,),device=predicted_landmarks.device)
    # weights[:17] = 1
    # weights[5:7] = 1
    # weights[10:12] = 1
    real_2d = landmarks_gt
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


# def eye_dis(landmarks):
#     # left eye:  [38,42], [39,41] - 1
#     # right eye: [44,48], [45,47] -1
#     eye_up = landmarks[:,[37, 38, 43, 44], :]
#     eye_bottom = landmarks[:,[41, 40, 47, 46], :]
#     dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
#     return dis

# new eye loss
def eye_dis(landmarks):
    # # left eye:  [38,42], [39,41] - 1
    # # right eye: [44,48], [45,47] -1
    # eye_up = landmarks[:,[37, 38, 43, 44], :]
    # eye_bottom = landmarks[:,[41, 40, 47, 46], :]
    # tx = [21, 22, 23, 31, 25, 29, 37, 38, 41, 45, 39, 47]
    # dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
    #
    # f = 2*torch.sqrt(((landmarks[:,[36,36,42,42]] - landmarks[:,[39,39,45,45], :])**2).sum(2))
    # # print(dis, f)
    # tx = [21, 22, 23, 31, 25, 29, 37, 38, 41, 45, 39, 47]
    eye_up = landmarks[:,[23, 25, 41, 39], :]
    eye_bottom = landmarks[:,[31, 29, 45, 47], :]
    dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]

    f = 2*torch.sqrt(((landmarks[:,[21,21,38,38]] - landmarks[:,[22,22,37,37], :])**2).sum(2))
    # print(dis, f)
    return dis/f

def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    real_2d = landmarks_gt
    pred_eyed = eye_dis(predicted_landmarks[:,:,:2])
    gt_eyed = eye_dis(real_2d[:,:,:2])

    loss = (pred_eyed - gt_eyed).abs().sum()/2

    return loss
#
#
#[276 282 283 285 293 295 296 300 334 336  46  52  53  55  63  65  66  70
 # 105 107 249 263 362 373 374 380 381 382 384 385 386 387 388 390 398 466
 #   7  33 133 144 145 153 154 155 157 158 159 160 161 163 173 246 168   6
 # 197 195   5   4 129  98  97   2 326 327 358   0  13  14  17  37  39  40
 #  61  78  80  81  82  84  87  88  91  95 146 178 181 185 191 267 269 270
 # 291 308 310 311 312 314 317 318 321 324 375 402 405 409 415]
def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1
    # lip_up = landmarks[:,[61, 62, 63], :]
    # lip_down = landmarks[:,[67, 66, 65], :]
    # kk = [61, 291, 78, 308, 185, 40, 39, 37, 0, 267, 269, 270, 409,      191, 80, 81, 82, 13, 312, 311, 310, 415,    95, 88, 178, 87, 14, 317, 402, 318, 324,   146, 91, 181, 84, 17, 314, 405, 321, 375]
    # t = [72, 90, 73, 91,     85, 71, 70, 69, 65, 87, 88, 89, 103,        86, 74, 75, 76, 66, 94, 93, 92, 104,     81, 79, 83, 78, 67, 96, 101, 97, 99,           82, 80, 84, 77, 68, 95, 102, 98, 100]

    lip_right = landmarks[:,[65, 72, 73,  85, 71, 70, 69, 87, 88, 89, 103,    86, 74, 75, 76, 66, 94, 93, 92, 104], :]
    lip_left = landmarks[:,[61, 90, 91,    82, 80, 84, 77, 95, 102, 98, 100,   81, 79, 83, 78, 67, 96, 101, 97, 99],:]
    # lip_up = landmarks[:,[49, 50, 51,52,53, 61,62,63], :]
    # lip_down = landmarks[:,[59, 58, 57, 56, 55, 67, 66, 65], :]
    # dis = torch.sqrt(((lip_up - lip_down)**2).sum(2)) #[bz, 4]
    dis = torch.sqrt(((lip_right - lip_left)**2).sum(2)) #[bz, 4]
    return dis

def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt).cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], landmarks_gt.shape[1], 1)).cuda()], dim=-1)
    real_2d = landmarks_gt
    pred_lipd = lip_dis(predicted_landmarks[:,:,:2])
    gt_lipd = lip_dis(real_2d[:,:,:2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss
# new one for mouth
def rel_dis(landmarks):
    # lip_right = landmarks[:,[57, 51, 48, 60, 61, 62, 63], :]
    # lip_left = landmarks[:,[8, 33, 54, 64, 67, 66, 65],:]
    lip_right = landmarks[:,[57, 51, 48, 60, 61, 62, 63, 49, 50, 51, 52, 53], :]
    lip_left = landmarks[:,[8, 33, 54, 64, 67, 66, 65, 59, 58, 57, 56, 55],:]

    dis = torch.sqrt(((lip_right - lip_left)**2).sum(2)) #[bz, 4]
    return dis
#
def relative_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # if torch.is_tensor(landmarks_gt) is not True:
    #     real_2d = torch.cat(landmarks_gt)#.cuda()
    # else:
    #     real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0],  landmarks_gt.shape[1], 1)).to(device=predicted_landmarks.device) #.cuda()
    #                          ], dim=-1)
    real_2d = landmarks_gt
    pred_lipd = rel_dis(predicted_landmarks[:, :, :2])
    gt_lipd = rel_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    # loss = F.mse_loss(pred_lipd, gt_lipd)

    return loss.mean()

def weighted_au_landmark_loss(predicted_landmarks, landmarks_gt, au, au_weight, weight=5.):
    real_2d = landmarks_gt.clone()
    real_2d[:,:,2] = 1.0

    # print('weight...',real_2d.shape)
    weights = torch.ones((landmarks_gt.shape[0],landmarks_gt.shape[1],), device=real_2d.device)


    # AU - landmark part
    for i in range(landmarks_gt.shape[0]):
        if True in (0.5 <= au[i,au_weight.brow_au]):
            weights[i,au_weight.brow] = weight / len(au_weight.brow)

        if True in (0.5 <= au[i,au_weight.brow_inner_au]):
            weights[i,au_weight.brow_inner] = weight / len(au_weight.brow_inner)

        if True in (0.5 <= au[i,au_weight.brow_outer_au]):
            weights[i,au_weight.brow_outer] = weight / len(au_weight.brow_outer)

        if True in (0.5 <= au[i,au_weight.eye_up_au]):
            weights[i,au_weight.eye_up] = weight / len(au_weight.eye_up)

        if True in (0.5 <= au[i,au_weight.eye_low_au]):
            weights[i,au_weight.eye_low] = weight / len(au_weight.eye_low)

        if True in (0.5 <= au[i,au_weight.eye_all_au]):
            weights[i,au_weight.eye_all] = weight / len(au_weight.eye_all)

        if True in (0.5 <= au[i,au_weight.nose_au]):
            weights[i,au_weight.nose] = weight / len(au_weight.nose)

        if True in (0.5 <= au[i,au_weight.lip_up_au]):
            weights[i,au_weight.lip_up] = weight / len(au_weight.lip_up)

        if True in (0.5 <= au[i,au_weight.lip_end_au]):
            weights[i,au_weight.lip_end] = weight / len(au_weight.lip_end)

        if True in (0.5 <= au[i,au_weight.mouth_au]):
            weights[i,au_weight.mouth] = weight / len(au_weight.mouth)

    weights[:,au_weight.lip_out] /= 5.0

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight

def related_au_landmark_loss(predicted_landmarks, landmarks_gt, au_weights, weight=1.):
    # print('real_2d_2',real_2d.shape)
    # weights = torch.ones((68,),device=predicted_landmarks.device)
    real_2d = landmarks_gt[:,:,:2]
    gt_distances = au_weights.au_related_landmark_distance(real_2d)
    pred_distances = au_weights.au_related_landmark_distance(predicted_landmarks)
    loss = 0.
    for i in range(len(gt_distances)):
        loss += (gt_distances[i] - pred_distances[i]).abs().mean()
    return loss

    # lip_right = landmarks[:,[65, 72, 73,  85, 71, 70, 69, 87, 88, 89, 103,    86, 74, 75, 76, 66, 94, 93, 92, 104], :]
    # lip_left = landmarks[:,[61, 90, 91,    82, 80, 84, 77, 95, 102, 98, 100,   81, 79, 83, 78, 67, 96, 101, 97, 99],:]
    
    # dis = torch.sqrt(((lip_right - lip_left)**2).sum(2)) #[bz, 4]

# new one
def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    real_2d = landmarks_gt

    # print('weight...',real_2d.shape)
    weights = torch.ones((landmarks_gt.shape[1],), device=real_2d.device)


    # nose points
    weights[52:65] = 1.5
    weights[57] = 3 #version 6 is 3
    weights[58] = 3
    weights[61] = 3
    weights[64] = 3
    # inner mouth
    weights[72:105] = 3
    weights[73:77] = 6 #version 6 is 8
    weights[91:95] = 6
    weights[86] = 6
    weights[66] = 6
    weights[104] = 6
    weights[72:74] = 8
    weights[90:92] = 8
    # weights[104] = 6


    weights[87] = 3
    weights[79] = 3
    weights[83] = 3
    weights[78] = 3
    weights[67] = 3
    weights[96] = 3
    weights[101] = 3
    weights[97] = 3
    weights[99] = 3
    # weights[48:60] = 1.5
    # weights[72] = 3
    # weights[73] = 3
    # weights[90] = 3

    # # no eye
    # weights[36:48] = 0

    # weights[36:48] = 3
    # weights[37] = 3
    # weights[38] = 3
    # weights[21] = 3
    # weights[22] = 3
    weights[20:52] = 8

    # eyebrow
    # weights[0] = 3
    # weights[7] = 3
    # weights[3] = 3
    # weights[9] = 3
    #
    # weights[13] = 3
    # weights[19] = 3
    # weights[10] = 3
    # weights[17] = 3

    weights[:20] = 8



    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight
# def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
#     #smaller inner landmark weights
#     # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
#     # import ipdb; ipdb.set_trace()
#     real_2d = landmarks_gt
#     weights = torch.ones((68,)).cuda()
#     weights[5:7] = 2
#     weights[10:12] = 2
#     # nose points
#     weights[27:36] = 1.5
#     weights[30] = 3
#     weights[31] = 3
#     weights[35] = 3
#     # inner mouth
#     weights[60:68] = 1.5
#     weights[48:60] = 1.5
#     weights[48] = 3
#     weights[54] = 3
#
#     loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
#     return loss_lmk_2d * weight

def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight


def ring_loss(ring_outputs, ring_type, margin, weight=1.):
    """
        computes ring loss for ring_outputs before FLAME decoder
        Inputs:
            ring_outputs = a list containing N streams of the ring; len(ring_outputs) = N
            Each ring_outputs[i] is a tensor of (batch_size X shape_dim_num)
            Aim is to force each row (same subject) of each stream to produce same shape
            Each row of first N-1 strams are of the same subject and
            the Nth stream is the different subject
        """
    tot_ring_loss = (ring_outputs[0]-ring_outputs[0]).sum()
    if ring_type == '51':
        diff_stream = ring_outputs[-1]
        count = 0.0
        for i in range(6):
            for j in range(6):
                pd = (ring_outputs[i] - ring_outputs[j]).pow(2).sum(1)
                nd = (ring_outputs[i] - diff_stream).pow(2).sum(1)
                tot_ring_loss = torch.add(tot_ring_loss,
                                (torch.nn.functional.relu(margin + pd - nd).mean()))
                count += 1.0

    elif ring_type == '33':
        perm_code = [(0, 1, 3),
                    (0, 1, 4),
                    (0, 1, 5),
                    (0, 2, 3),
                    (0, 2, 4),
                    (0, 2, 5),
                    (1, 0, 3),
                    (1, 0, 4),
                    (1, 0, 5),
                    (1, 2, 3),
                    (1, 2, 4),
                    (1, 2, 5),
                    (2, 0, 3),
                    (2, 0, 4),
                    (2, 0, 5),
                    (2, 1, 3),
                    (2, 1, 4),
                    (2, 1, 5)]
        count = 0.0
        for i in perm_code:
            pd = (ring_outputs[i[0]] - ring_outputs[i[1]]).pow(2).sum(1)
            nd = (ring_outputs[i[1]] - ring_outputs[i[2]]).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss,
                            (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0

    tot_ring_loss = (1.0/count) * tot_ring_loss

    return tot_ring_loss * weight


######################################## images/features/perceptual
def gradient_dif_loss(prediction, gt):
    prediction_diff_x =  prediction[:,:,1:-1,1:] - prediction[:,:,1:-1,:-1]
    prediction_diff_y =  prediction[:,:,1:,1:-1] - prediction[:,:,1:,1:-1]
    gt_x =  gt[:,:,1:-1,1:] - gt[:,:,1:-1,:-1]
    gt_y =  gt[:,:,1:,1:-1] - gt[:,:,:-1,1:-1]
    diff = torch.mean((prediction_diff_x-gt_x)**2) + torch.mean((prediction_diff_y-gt_y)**2)
    return diff.mean()


def get_laplacian_kernel2d(kernel_size: int):
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d

def laplacian_hq_loss(prediction, gt):
    # https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
    b, c, h, w = prediction.shape
    kernel_size = 3
    kernel = get_laplacian_kernel2d(kernel_size).to(prediction.device).to(prediction.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    lap_pre = F.conv2d(prediction, kernel, padding=padding, stride=1, groups=c)
    lap_gt = F.conv2d(gt, kernel, padding=padding, stride=1, groups=c)

    return ((lap_pre - lap_gt)**2).mean()


class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x / self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out

##
# class VGG19FeatLayer(nn.Module):
#     def __init__(self):
#         super(VGG19FeatLayer, self).__init__()
#         self.vgg19 = models.vgg19(pretrained=True).features.cuda().eval()
#         ## WHY ARE THE CONSTANTS SET THIS WAY
#         # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) #.cuda()
#         # self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) #.cuda()
#         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda() )
#         self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda() )


#     def forward(self, x):
#         out = {}
#         x = x - self.mean
#         x = x / self.std
#         ci = 1
#         ri = 0
#         for layer in self.vgg19.children():
#             if isinstance(layer, nn.Conv2d):
#                 ri += 1
#                 name = 'conv{}_{}'.format(ci, ri)
#             elif isinstance(layer, nn.ReLU):
#                 ri += 1
#                 name = 'relu{}_{}'.format(ci, ri)
#                 layer = nn.ReLU(inplace=False)
#             elif isinstance(layer, nn.MaxPool2d):
#                 ri = 0
#                 name = 'pool_{}'.format(ci)
#                 ci += 1
#             elif isinstance(layer, nn.BatchNorm2d):
#                 name = 'bn_{}'.format(ci)
#             else:
#                 raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
#             x = layer(x)
#             out[name] = x
#         # print([x for x in out])
#         return out


class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        scaled_dist = torch.clamp(scaled_dist, max=8.872e+01)
        dist_before_norm = torch.exp(((self.bias - scaled_dist)/self.nn_stretch_sigma).clamp(min=-10.0, max=10.0))
        # dist_before_norm = torch.exp(((self.bias - scaled_dist)/self.nn_stretch_sigma))#.clamp(min=-35.0, max=35.0))
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content
        # print(self.style_loss + self.content_loss)
        return self.style_loss + self.content_loss


# class IDMRFLoss(nn.Module):
#     def __init__(self, featlayer=VGG19FeatLayer):
#         super(IDMRFLoss, self).__init__()
#         self.featlayer = featlayer()
#         self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
#         self.feat_content_layers = {'relu4_2': 1.0}
#         self.bias = 1.0
#         self.nn_stretch_sigma = 0.5
#         self.lambda_style = 1.0
#         self.lambda_content = 1.0

#     def sum_normalize(self, featmaps):
#         reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
#         return featmaps / reduce_sum

#     def patch_extraction(self, featmaps):
#         patch_size = 1
#         patch_stride = 1
#         patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
#         self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
#         dims = self.patches_OIHW.size()
#         self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
#         return self.patches_OIHW

#     def compute_relative_distances(self, cdist): # here I think the problem occurs
#         epsilon = 1e-5
#         div = torch.min(cdist, dim=1, keepdim=True)[0]
#         #print("\n\n***div is ")
#         #print(div)
#         #print("\n\n**cdist is ")
#         #print(cdist)
#         relative_dist = cdist / (div + epsilon)
#         #print(relative_dist)
#         return relative_dist

#     def exp_norm_relative_dist(self, relative_dist):
#         scaled_dist = relative_dist
        
#         #print(self.bias-scaled_dist) # -4.123e+05 , 7e+03
#         #print(self.nn_stretch_sigma) # 0.5
        
#         # dist_before_exp = (self.bias - scaled_dist)/self.nn_stretch_sigma
#         # dist_befroe_exp_after_norm = self.sum_normalize(dist_before_exp)
#         # dist_before_norm = torch.exp(dist_befroe_exp_after_norm)
        
#         #new
#         # tmp = scaled_dist
#         # relu = torch.nn.ReLU()
#         # tmp2 = relu(tmp)
#         # tmp3 = (self.bias - tmp2)/self.nn_stretch_sigma
#         # dist_before_norm = torch.exp(tmp3) 
#         scaled_dist = torch.clamp(scaled_dist, max=8.872e+01)
#         dist_before_norm = torch.exp(((self.bias - scaled_dist)/self.nn_stretch_sigma).clamp(min=-10.0, max=10.0))
        
#         # dist_before_norm = torch.exp((self.bias - scaled_dist)/self.nn_stretch_sigma) #original 
#         # infinity 값이 있는지 확인
#         # has_inf = torch.isinf(dist_before_norm).any().item()  # tensor에 infinity 값이 있는지 확인하고, Boolean 값을 반환합니다.

#         # if has_inf:
#         #     print("텐서 내에 infinity 값이 있습니다.")
#         #     tmp= input("debug go")
#         # else:
#         #     print("텐서 내에 infinity 값이 없습니다.")

#         #dist_before_norm =(self.bias - scaled_dist)/self.nn_stretch_sigma
#         #print(dist_before_norm)
#         self.cs_NCHW = self.sum_normalize(dist_before_norm)
#         return self.cs_NCHW

#     def mrf_loss(self, gen, tar):
#         epsilon = 1e-8
#         meanT = torch.mean(tar, 1, keepdim=True)
#         gen_feats, tar_feats = gen - meanT, tar - meanT

#         gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
#         tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)
        
#         # print(torch.eq(gen_feats_norm,tar_feats_norm))

#         gen_normalized = gen_feats / gen_feats_norm
#         tar_normalized = tar_feats / tar_feats_norm

#         cosine_dist_l = []
#         BatchSize = tar.size(0)

#         for i in range(BatchSize):
#             tar_feat_i = tar_normalized[i:i+1, :, :, :]
#             gen_feat_i = gen_normalized[i:i+1, :, :, :]
#             patches_OIHW = self.patch_extraction(tar_feat_i)
            
#             cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            

#             # a = gen_feat_i - torch.norm(patches_OIHW)
#             # b = patches_OIHW - torch.norm(patches_OIHW)
#             # cosine_dist_i = 1 - F.conv2d(a, b)/(torch.norm(gen_feat_i, p=2) * torch.norm(patches_OIHW, p=2))
#             # cos_dist =  1 - F.cosine_similarity(gen_feat_i, patches_OIHW)
#             # a = cosine_dist_i
#             # check = torch.where(a < 0.0, True, False) 
#             #print(torch.masked_select(a,check))
#             #cosine_dist_i[cosine_dist_i < 0] = 0
#             #print(torch.masked_select(cosine_dist_i,check))
#             cosine_dist_l.append(cosine_dist_i)
#         cosine_dist = torch.cat(cosine_dist_l, dim=0)
#         #print(cosine_dist)
#         # check = torch.where(cosine_dist < 0.0, True, False) # 조건에 맞으면 True, 아니면 False
#         #print(check)    
        

#         # # 음수값이 있는지 확인
#         # has_negative = (cosine_dist < 0).any().item()  # tensor에 음수값이 있는지 확인하고 Boolean 값을 반환합니다.
        
#         # if has_negative:
#         #     print("텐서 내에 음수값이 존재합니다.")
            
#         #     #negative value counts
#         #     num_negatives = (cosine_dist < 0).sum().item()  # tensor 내의 음수값의 개수를 세고, 이를 정수형으로 반환합니다.

#         #     print("텐서 내에 {}개의 음수값이 있습니다.".format(num_negatives))
#         #     # 음수값 추출
#         #     # negative_values = cosine_dist[cosine_dist < 0].tolist()  # 음수값을 추출하고, 리스트로 변환합니다.
#         #     # print("텐서 내의 음수값:", negative_values)
#         # else:
#         #     print("텐서 내에 음수값이 존재하지 않습니다.")
        
#         # # 최대값과 최소값 출력
#         # max_value = torch.max(cosine_dist).item()  # 텐서 내의 최대값을 추출하고, 이를 정수형으로 변환합니다.
#         # min_value = torch.min(cosine_dist).item()  # 텐서 내의 최소값을 추출하고, 이를 정수형으로 변환합니다.
#         # avg_value = torch.mean(cosine_dist).item()
#         # print("텐서의 최대값:", max_value)
#         # print("텐서의 최소값:", min_value)
#         # print("tensor's avg:", avg_value)

#         cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
#         relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        
#         # has_88 = (relative_dist > 88).any().item()  # tensor에 음수값이 있는지 확인하고 Boolean 값을 반환합니다.

#         # max_value = torch.max(relative_dist).item()
#         # print(max_value)

        
#         rela_dist = self.exp_norm_relative_dist(relative_dist) #exp (relative_dist) -> some of elements go inf

#         # max_value = torch.max(rela_dist).item()
#         # print("rela_dist",max_value)
#         # rela_dist = torch.nan_to_num(rela_dist)
#         # max_value = torch.max(rela_dist).item()
#         # print(max_value)
#         # max_value = torch.min(rela_dist).item()
#         # print("min of rela_dist",max_value)

#         dims_div_mrf = rela_dist.size()
        
#         # max_value = torch.max(dims_div).item()
#         # print(max_value)

#         k_max_nc = torch.Tensor.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0] # k_max_nc -> nan
#         div_mrf = torch.mean(k_max_nc, dim=1)
#         div_mrf_sum = -torch.log(div_mrf+epsilon)
#         div_mrf_sum = torch.sum(div_mrf_sum)
#         return div_mrf_sum

#     def forward(self, gen, tar):
#         gen_vgg_feats = self.featlayer(gen)
#         tar_vgg_feats = self.featlayer(tar)

#         style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
#         self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

#         content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
#         self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

#         return self.style_loss + self.content_loss
    
#     def train(self, b = True):
#         # there is nothing trainable about this loss
#         return super().train(False)

# class IDMRFLoss(nn.Module):
#     def __init__(self, featlayer=VGG19FeatLayer):
#         super(IDMRFLoss, self).__init__()
#         self.featlayer = featlayer()
#         self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
#         self.feat_content_layers = {'relu4_2': 1.0}
#         self.bias = 1.0
#         self.nn_stretch_sigma = 0.5
#         self.lambda_style = 1.0
#         self.lambda_content = 1.0

#     def sum_normalize(self, featmaps):
#         reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
#         return featmaps / reduce_sum

#     def patch_extraction(self, featmaps):
#         patch_size = 1
#         patch_stride = 1
#         patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
#         self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
#         dims = self.patches_OIHW.size()
#         self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
#         return self.patches_OIHW

#     def compute_relative_distances(self, cdist):
#         epsilon = 1e-5
#         div = torch.min(cdist, dim=1, keepdim=True)[0]
#         relative_dist = cdist / (div + epsilon)
#         return relative_dist

#     def exp_norm_relative_dist(self, relative_dist):
#         scaled_dist = relative_dist
#         dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
#         self.cs_NCHW = self.sum_normalize(dist_before_norm)
#         return self.cs_NCHW

#     def mrf_loss(self, gen, tar):
#         meanT = torch.mean(tar, 1, keepdim=True)
#         gen_feats, tar_feats = gen - meanT, tar - meanT

#         gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
#         tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

#         gen_normalized = gen_feats / gen_feats_norm
#         tar_normalized = tar_feats / tar_feats_norm

#         cosine_dist_l = []
#         BatchSize = tar.size(0)

#         for i in range(BatchSize):
#             tar_feat_i = tar_normalized[i:i + 1, :, :, :]
#             gen_feat_i = gen_normalized[i:i + 1, :, :, :]
#             patches_OIHW = self.patch_extraction(tar_feat_i)

#             cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
#             cosine_dist_l.append(cosine_dist_i)
#         cosine_dist = torch.cat(cosine_dist_l, dim=0)
#         cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
#         relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
#         rela_dist = self.exp_norm_relative_dist(relative_dist)
#         dims_div_mrf = rela_dist.size()
#         k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
#         div_mrf = torch.mean(k_max_nc, dim=1)
#         div_mrf_sum = -torch.log(div_mrf)
#         div_mrf_sum = torch.sum(div_mrf_sum)
#         return div_mrf_sum

#     def forward(self, gen, tar):
#         ## gen: [bz,3,h,w] rgb [0,1]
#         gen_vgg_feats = self.featlayer(gen)
#         tar_vgg_feats = self.featlayer(tar)
#         style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
#                            layer in self.feat_style_layers]
#         self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style

#         content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
#                              for layer in self.feat_content_layers]
#         self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content

#         return self.style_loss + self.content_loss

#         # loss = 0
#         # for key in self.feat_style_layers.keys():
#         #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
#         # return loss

#     def train(self, b = True):
#         # there is nothing trainable about this loss
#         return super().train(False)

######################################################## vgg16 face

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        self.mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])/255.).float().view(1, 3, 1, 1).cuda()
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        out = {}
        x = x - self.mean
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        out['relu3_2'] = x
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        out['relu4_2'] = x
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc8(x)
        out['last'] = x
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG_16().float()
        self.featlayer.load_weights(path="data/face_recognition_model/vgg_face_torch/VGG_FACE.t7")
        self.featlayer = self.featlayer.cuda().eval()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW
    # detial
    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss
        # loss = 0
        # for key in self.feat_style_layers.keys():
        #     loss += torch.mean((gen_vgg_feats[key] - tar_vgg_feats[key])**2)
        # return loss

##############################################
## ref: https://github.com/cydonia999/VGGFace2-pytorch
from ..models.frnet import resnet50, load_state_dict
class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_model, pretrained_data='vggface2'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().cuda()
        load_state_dict(self.reg_model, pretrained_model)
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()

    # !!!here???!
        # out = []
    def reg_features(self, x):
        margin=10
        # x = x[:,:,margin:224-margin,margin:224-margin]
        x = x[:,:,margin:448-margin,margin:448-margin]
        x = F.interpolate(x*2. - 1., [448,448], mode='bilinear')
        # x = F.interpolate(x*2. - 1., [224,224], mode='bilinear')
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

        # import ipdb; ipdb.set_trace()
        # x = F.interpolate(x*2. - 1., [224,224], mode='nearest')

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2,1,0], :, :].permute(0,2,3,1) * 255 - self.mean_bgr
        img = img.permute(0,3,1,2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss


ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class

def FFLoss(gen, tar):
    return ffl(gen, tar)    

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        IS_HIGH_VERSION=True
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            # matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

