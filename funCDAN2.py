#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:20:39 2020

@author: user1
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import alexnet
import numpy as np
cos1=nn.CosineSimilarity(dim=1,eps=1e-6)
def ContLoss(inputX1,inputX2):
    cosSim1=cos1(inputX1,inputX2)
    Temper1=cosSim1.mean()
    size1=inputX1.size(0)
    closs1=0
    for ii1 in range(size1):
        fenzi1=torch.exp(cosSim1[ii1]/Temper1)
        fenmu1 =0
        for jj1 in range(size1):
            if ii1 != jj1:
                fenmu1 += torch.exp(cosSim1[ii1]/Temper1)
        closs1 += fenzi1/fenmu1
    return -1.0*torch.log(closs1)/size1  
def ContLoss2(inputX1,inputX2,lbY):
    cosSim1=cos1(inputX1,inputX2)
    Temper1=cosSim1.mean()
    sz1=inputX1.size(0)
    closs1=0
    pos_mask=(lbY.expand(sz1,sz1).eq(lbY.expand(sz1,sz1).t())).to(torch.float32).cuda()-torch.eye(sz1,dtype=torch.float32).cuda()
    neg_mask=torch.ones((sz1,sz1),dtype=torch.float32).cuda()-pos_mask-torch.eye(sz1,dtype=torch.float32).cuda()
    for ii1 in range(sz1):
        fr1=(cosSim1.mul(pos_mask[ii1].to(torch.float32))/Temper1).sum() +0.000001
        fr2=(cosSim1.mul(neg_mask[ii1].to(torch.float32))/Temper1).sum() +0.000001
        closs1 += -1.0*torch.log(fr1/fr2)
    return closs1/sz1
def CosSim1(inputX1,inputX2):
    sz1=inputX1.size(0)
    CosMt1=torch.zeros((sz1,sz1))
    for i1 in range(sz1):
        for j1 in range(sz1):
            CosMt1[i1,j1]=cos1(inputX1[i1].unsqueeze(0),inputX2[j1].unsqueeze(0))
    return CosMt1
def softNNLoss3(inputX1,lbY):
    inputX1=inputX1.cuda()
    lbY=lbY.cuda()
    cosSimMt1=CosSim1(inputX1,inputX1).cuda()
    Temper1=cosSimMt1.mean()
    sz1=inputX1.size(0)
    pos_mask=(lbY.expand(sz1,sz1).eq(lbY.expand(sz1,sz1).t())).to(torch.float32).cuda()-torch.eye(sz1,dtype=torch.float32).cuda()
    neg_mask=torch.ones((sz1,sz1),dtype=torch.float32).cuda()-pos_mask-torch.eye(sz1,dtype=torch.float32).cuda()
    fr1=(torch.exp(cosSimMt1.mul(pos_mask.to(torch.float32))/Temper1)-torch.ones((sz1,sz1),dtype=torch.float32).cuda()).sum(dim=1).cuda()+0.000001
    fr2=(torch.exp(cosSimMt1.mul(neg_mask.to(torch.float32))/Temper1)--torch.ones((sz1,sz1),dtype=torch.float32).cuda()).sum(dim=1).cuda()+0.000001
    return -1.0*torch.log(fr1/fr2).mean()    
def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors)) #
    D += vectors.pow(2).sum(dim=1).view(1, -1) #sum by line
    D += vectors.pow(2).sum(dim=1).view(-1, 1)
    return D
def softNNLoss(feaX,lbY):
    sz1=feaX.shape[0]
    distMat=pdist(feaX).cuda()
    mean_batch=distMat.mean().cuda()
    pos_mask=(lbY.expand(sz1,sz1).eq(lbY.expand(sz1,sz1).t())).to(torch.float32).cuda()-torch.eye(sz1,dtype=torch.float32).cuda()
    neg_mask=torch.ones((sz1,sz1),dtype=torch.float32).cuda()-pos_mask-torch.eye(sz1,dtype=torch.float32).cuda()
    fr1=torch.exp(-1.0*distMat.mul(pos_mask.to(torch.float32))/mean_batch).sum(dim=1)+0.000001
    fr2=torch.exp(-1.0*distMat.mul(neg_mask.to(torch.float32))/mean_batch).sum(dim=1)+0.000001
    return -1.0*torch.log(fr1/fr2).mean()
def softNNLoss2(feaX,lbY):
    sz1=feaX.shape[0]
    distMat=pdist(feaX).cuda()
    mean_batch=distMat.mean().cuda()
    pos_mask=(lbY.expand(sz1,sz1).eq(lbY.expand(sz1,sz1).t())).to(torch.float32).cuda()-torch.eye(sz1,dtype=torch.float32).cuda()
    neg_mask=torch.ones((sz1,sz1),dtype=torch.float32).cuda()-pos_mask-torch.eye(sz1,dtype=torch.float32).cuda()
    fr1=torch.exp(-1.0*distMat.mul(pos_mask.to(torch.float32))/mean_batch).sum(dim=1)+0.000001
    fr2=torch.exp(-1.0*distMat/mean_batch).sum(dim=1)+0.000001
    return -1.0*torch.log(fr1/fr2).mean()
tfs_tr1=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

tfs_te1=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
from torchvision import transforms
import torchvision.transforms.functional as TransF
import random
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TransF.rotate(x, angle)
rotation_transform = MyRotationTransform(angles=[-115,-25, 0, 25, 115])
Crop_tfs2=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation((20,130)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def get_color_distortion(s=1.0):
    color_jitter=transforms.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)
    rnd_color_jitter=transforms.RandomApply([color_jitter],p=0.8)
    rnd_gray=transforms.RandomGrayscale(p=0.2)
    color_distort=transforms.Compose([
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return color_distort
rand_color=transforms.Compose([
        transforms.ColorJitter(0.8,0.8,0.8,0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
def calc_coeff2(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=8000.0):
    return np.float(2.4 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

        
class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(AlexNetFc, self).__init__()
    model_alexnet = alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(4096, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_alexnet.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
class AdversarialNetwork2(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork2, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    xf1 = self.dropout2(x)
    y = self.ad_layer3(xf1)
    y = self.sigmoid(y)
    return y,xf1

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer

def EntropyLoss1(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


