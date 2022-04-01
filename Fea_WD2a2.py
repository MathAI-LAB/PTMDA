#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract  feature
"""
import os
os.chdir('/data2/yonghui_/Bn_Spur1/')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms,models
import torch.nn as nn
import torch.optim as optim
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list2 import ImageList_SV1,ImageList
##parameters,

dir_fea='./Pred1/'
BSZ=30
###Load  data
config={}
prep_dict = {}#'prep': 
config["prep"]={'test_10crop': True,'params':{'resize_size': 256,'crop_size':224,'alexnet': False}}
prep_config = {'test_10crop': True,'params':{'resize_size': 256,'crop_size':224,'alexnet': False}}
prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])#返回10个 transforms.Compose,
## prepare data
dsets = {}
dset_loaders = {}
data_config={}
dsets["test"] = [ImageList(open('../data1_/office/amazon_list.txt').readlines(), \
                               transform=prep_dict["test"][i]) for i in range(10)]# 10个Dataset类的列表,长度=10
dset_loaders["test"] = [DataLoader(dset, batch_size=BSZ, \
                                       shuffle=False, num_workers=2) for dset in dsets['test']]

base_network=torch.load('/data2/yonghui_/Bn_Spur1/WD2a1/WD2a3_Rs50_8721.pth')
base_network=base_network.cuda()

start_test = True
with torch.no_grad():
    iter_test = [iter(dset_loaders['test'][i]) for i in range(10)]
    for i in range(len(dset_loaders['test'][0])):
        data = [iter_test[j].next() for j in range(10)]
        inputs = [data[j][0] for j in range(10)]
        labels = data[0][1]
        #path1=data[0][2]
        for j in range(10):
            inputs[j] = inputs[j].cuda()
        labels = labels
        outputs = []
        outputs1 = []
        outputs2 = []
        for j in range(10):
            _, predict_out1 = base_network(inputs[j], source=1)
            outputs1.append(nn.Softmax(dim=1)(predict_out1))
            
            _, predict_out2 = base_network(inputs[j], source=2)
            outputs2.append(nn.Softmax(dim=1)(predict_out2))
            
            predict_out = predict_out1 + predict_out2
            outputs.append(nn.Softmax(dim=1)(predict_out))
        outputs = sum(outputs)
        outputs1 = sum(outputs1)
        outputs2 = sum(outputs2)
        if start_test:
            all_output = outputs.float().cpu()
            all_label = labels.float()
            
            all_output1 = outputs1.float().cpu()
            
            all_output2 = outputs2.float().cpu()
            
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.float().cpu()), 0)
            
            all_output1 = torch.cat((all_output1, outputs1.float().cpu()), 0)
            
            all_output2 = torch.cat((all_output2, outputs2.float().cpu()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
    max0, predict = torch.max(all_output, 1)
    max1, predict1 = torch.max(all_output1, 1)
    max2, predict2 = torch.max(all_output2, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    accuracy1 = torch.sum(torch.squeeze(predict1).float() == all_label).item() / float(all_label.size()[0])
    accuracy2 = torch.sum(torch.squeeze(predict2).float() == all_label).item() / float(all_label.size()[0])
    print(max1.shape)
    print(predict1.shape)
    print(all_label.shape)
    save1=torch.cat((max1.unsqueeze(1),predict1.float().unsqueeze(1),all_label.unsqueeze(1)),dim=1)
save1=save1.cpu().numpy()
np.savetxt(os.path.join(dir_fea,'WD2a3_Rs50_8721.txt'),save1,fmt='%.6f',delimiter=' ')
print('Acc0',accuracy,'Acc1',accuracy1,'Acc2',accuracy2)
#fea_np2=fea_all2.cpu().numpy()
#np.savetxt(os.path.join(dir_fea,'TryBestAC_3647.csv'),fea_np2[1:],fmt='%.6f',delimiter=',')
#np.savetxt(os.path.join(dir_fea,'TryBestAC_3647.txt'),fea_np2[1:],fmt='%.6f',delimiter=' ')
        
        
    
