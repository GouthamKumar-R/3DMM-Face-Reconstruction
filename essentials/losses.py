import numpy as np
import torch
import torch.nn.functional as F

def photo_loss(pred_img, gt_img, img_mask):
    pred_img = pred_img.float()
    loss = torch.sqrt(torch.sum(torch.square(pred_img - gt_img), 3))*img_mask/255
    loss = torch.sum(loss, dim=(1, 2)) / torch.sum(img_mask, dim=(1, 2))
    loss = torch.mean(loss)

    return loss


def lm_loss(pred_lms, gt_lms, img_size=224):
    w = torch.ones((1, 68)).to(pred_lms.device)
    # we set higher weights for landmarks around the mouth and nose regions
    # landmark_weight = tf.concat([tf.ones([1,28]),20*tf.ones([1,3]),tf.ones([1,29]),20*tf.ones([1,8])],axis = 1)
    # landmark_weight = tf.tile(landmark_weight,[tf.shape(landmark_p)[0],1])
    w[:, 28:31] = 10
    w[:, 48:68] = 10
    norm_w = w / w.sum()
    loss = torch.sum(torch.square(pred_lms/img_size - gt_lms/img_size), dim=2) * norm_w
    loss = torch.mean(loss.sum(1))

    return loss

def reg_loss(id_coeff, ex_coeff, tex_coeff):

    loss = torch.square(id_coeff).sum() + \
            torch.square(tex_coeff).sum() * 1.7e-3 + \
            torch.square(ex_coeff).sum(1).mean() * 0.8

    return loss
