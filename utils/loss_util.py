import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter
import os



class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1, out2, out3, gt1, in_img1, isloss4, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        in_img2 = F.interpolate(in_img1, scale_factor=0.5 , mode='bilinear', align_corners=False)
        in_img3 = F.interpolate(in_img1, scale_factor=0.25, mode='bilinear', align_corners=False)
        # print("L1 Loss = ", F.l1_loss(out1, gt1))
        # print("VGG Loss = ", self.loss_fn(out1, gt1, feature_layers=feature_layers))
        # print("============================")
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt2)
        loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out3, gt3)

        # loss1 = self.lam*F.l1_loss(out1, in_img )
        # loss2 = self.lam*F.l1_loss(out2, in_img2)
        # loss3 = self.lam*F.l1_loss(out3, in_img3)

        if isloss4:
            E_in1  = torch.sqrt(torch.sum(torch.pow(in_img1.max(dim=1)[0], 2))/(in_img1.shape[2]*in_img1.shape[3]))
            E_out1 = torch.sqrt(torch.sum(torch.pow(   out1.max(dim=1)[0], 2))/(in_img1.shape[2]*in_img1.shape[3]))

            E_in2  = torch.sqrt(torch.sum(torch.pow(in_img2.max(dim=1)[0], 2))/(in_img1.shape[2]*in_img1.shape[3]))
            E_out2 = torch.sqrt(torch.sum(torch.pow(   out2.max(dim=1)[0], 2))/(in_img1.shape[2]*in_img1.shape[3]))

            E_in3  = torch.sqrt(torch.sum(torch.pow(in_img3.max(dim=1)[0], 2))/(in_img1.shape[2]*in_img1.shape[3]))
            E_out3 = torch.sqrt(torch.sum(torch.pow(   out3.max(dim=1)[0], 2))/(in_img1.shape[2]*in_img1.shape[3]))

            loss4 = F.l1_loss(E_in1, E_out1) + F.l1_loss(E_in2, E_out2) + F.l1_loss(E_in3, E_out3)
        
            return loss1 + loss2 + loss3 + 5*loss4
        else:
            return loss1 + loss2 + loss3

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
