#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
os.chdir('/data2/yonghui_/Bn_Spur1')
from logs1 import *
import sys
sys.stdout = Logger( 'WD2a_stp2_s2.txt')
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from funCDAN2 import *
from torch import optim
import loss
from network3 import ResNetFc2
from data_list2 import ImageList_SV2,ImageList
dir_model= './WD2a1'
Sour1='../data1_/office/webcam_list.txt'
Sour2='../data1_/office/dslr_list.txt'
Targ='../data1_/office/amazon_list.txt'
TargTr='./Aprd1.txt'
BSZs1= 36
BSZs2= 18
BSZtg= 18
device=torch.device("cuda:0")
class_num=31
iters=10000
lr_init=0.001
seed=8
bottleneck_dim=256
###############Read Data,
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

DTsetS1 = ImageList(open(Sour1).readlines(),transform=tfs_tr1)
LoaderS1 =DataLoader(DTsetS1,BSZs1,shuffle=True,drop_last=True, num_workers=5)

DTsetS2 = ImageList(open(Sour2).readlines(),transform=tfs_tr1)
LoaderS2 =DataLoader(DTsetS2,BSZs1,shuffle=True,drop_last=True, num_workers=5)

DTsetS12 = ImageList(open(Sour1).readlines(),transform=tfs_tr1)
LoaderS12 =DataLoader(DTsetS12,BSZs2,shuffle=True,drop_last=True, num_workers=5)

DTsetS22 = ImageList(open(Sour2).readlines(),transform=tfs_tr1)
LoaderS22 =DataLoader(DTsetS22,BSZs2,shuffle=True,drop_last=True, num_workers=5)

DTsetTGtr = ImageList_SV2(open(TargTr).readlines(),transform=tfs_tr1)
#print('DTsetTGtr**',len(DTsetTGtr[0]))
#print('DTsetTGtr0**',DTsetTGtr[0][0].shape)
#print('DTsetTGtr1**',DTsetTGtr[0][1].shape)
#print('DTsetTGtr1**',DTsetTGtr[0][2].shape)

LoaderTGtr =DataLoader(DTsetTGtr,BSZtg,shuffle=True,drop_last=True, num_workers=5)
#x1,x2,x3=next(iter(LoaderTGtr))
DTsetTGte =  ImageList(open(Targ).readlines(),transform=tfs_te1)
LoaderTGte =DataLoader(DTsetTGte,20,shuffle=False,drop_last=False, num_workers=5)

####
#base_network=ResNetFc2('ResNet50',use_bottleneck=True, bottleneck_dim=256,new_cls=True,class_num=12)
base_network=torch.load('/data2/yonghui_/Bn_Spur1/WD2a1/WD2a3_Rs50_8721.pth')
base_network=base_network.to(device)

ad_net = AdversarialNetwork(bottleneck_dim* class_num, 1024)
ad_net=ad_net.to(device)

ad_net2 = AdversarialNetwork(bottleneck_dim* class_num, 1024)
ad_net2=ad_net2.to(device)

parameter_list = base_network.get_parameters() + ad_net.get_parameters()
optimizer1 = optim.SGD(parameter_list,lr=lr_init,momentum=0.9,weight_decay= 0.0005,nesterov=True)

parameter_list2 = base_network.get_parameters() + ad_net2.get_parameters()
optimizer2 = optim.SGD(parameter_list2,lr=lr_init,momentum=0.9,weight_decay= 0.0005,nesterov=True)

source1_iter = iter(LoaderS1)
target_iter = iter(LoaderTGtr)
#print(len(LoaderTGtr))
#print(len(target_iter))
source2_iter = iter(LoaderS2)
source12_iter = iter(LoaderS12)
source22_iter = iter(LoaderS22)
best_acc = 0.0
best_idx=0
for i in range(iters):
    ## first trial, S1 --> S22 and Targ,
    base_network.train()
    ad_net.train()
    try:
        inputs_source, labels_source = source1_iter.next()
    except Exception as err:
        source1_iter = iter(LoaderS1)
        inputs_source, labels_source = source1_iter.next()
    try:
        inputs_source2, labels_source2 = source22_iter.next()
    except Exception as err:
        source22_iter = iter(LoaderS22)
        inputs_source2, labels_source2 = source22_iter.next()
    try:
        inputs_target,fake0,mask1= target_iter.next()
    except Exception as err:
        target_iter = iter(LoaderTGtr)
        inputs_target,fake0,mask1 = target_iter.next()
    optimizer = inv_lr_scheduler(optimizer1,i,lr= lr_init, gamma=0.001,power=0.75)
    optimizer.zero_grad()   
    
    inputs_target=inputs_target.to(device)
    mask1=mask1.float()
    mask1=mask1.to(device)
    fake0=fake0.squeeze()
    fake0=fake0.to(device)
    
    inputs_source=inputs_source.to(device)
    labels_source = labels_source.to(device)
    
    inputs_source2=inputs_source2.to(device)
    labels_source2 = labels_source2.to(device)
    
    features_source, outputs_source = base_network(inputs_source, source=1)
    features_source2, outputs_source2 = base_network(inputs_source2, source=1)
    features_target, outputs_target = base_network(inputs_target, source=1)
    
    features = torch.cat((features_source,features_source2, features_target), dim=0)
    outputs = torch.cat((outputs_source,outputs_source2, outputs_target), dim=0)
    softmax_out = nn.Softmax(dim=1)(outputs)
    
    entropy = loss.Entropy(softmax_out)
    transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy,
                              calc_coeff(i), random_layer=None)
    outputs2 = torch.cat((outputs_source,outputs_source2), dim=0)
    features2 = torch.cat((features_source,features_source2), dim=0)
    labels2 = torch.cat((labels_source,labels_source2), dim=0)
    classifier_loss = nn.CrossEntropyLoss()(outputs2, labels2)
    
    cs_loss_tgt1=nn.CrossEntropyLoss(reduction='none')(outputs_target,fake0)
    cs_loss_tgt2=(cs_loss_tgt1*mask1).mean()
    sfLoss1=softNNLoss(features2,labels2)
    total_loss = 1.0 * transfer_loss + classifier_loss+cs_loss_tgt2+calc_coeff2(i)*sfLoss1
    total_loss.backward()
    optimizer.step()
    
    ## Second trial, S2 --> S12 and Targ,
    base_network.train()
    ad_net2.train()
    try:
        inputs_source, labels_source = source2_iter.next()
    except Exception as err:
        source2_iter = iter(LoaderS2)
        inputs_source, labels_source = source2_iter.next()
    try:
        inputs_source2, labels_source2 = source12_iter.next()
    except Exception as err:
        source12_iter = iter(LoaderS12)
        inputs_source2, labels_source2 = source12_iter.next()
    try:
        inputs_target,fake0,mask1 = target_iter.next()
    except Exception as err:
        target_iter = iter(LoaderTGtr)
        inputs_target,fake0,mask1 = target_iter.next()
    optimizer2 = inv_lr_scheduler(optimizer2,i,lr=lr_init, gamma=0.001,power=0.75)
    optimizer2.zero_grad()
    inputs_target=inputs_target.to(device)
    mask1=mask1.float()
    mask1=mask1.to(device)
    fake0=fake0.squeeze()
    fake0=fake0.to(device)
    
    inputs_source=inputs_source.to(device)
    labels_source = labels_source.to(device)
    
    inputs_source2=inputs_source2.to(device)
    labels_source2 = labels_source2.to(device)
    
    features_source, outputs_source = base_network(inputs_source, source=2)
    features_source2, outputs_source2 = base_network(inputs_source2, source=2)
    features_target, outputs_target = base_network(inputs_target, source=2)
    
    features = torch.cat((features_source,features_source2, features_target), dim=0)
    outputs = torch.cat((outputs_source,outputs_source2, outputs_target), dim=0)
    softmax_out = nn.Softmax(dim=1)(outputs)
    
    entropy = loss.Entropy(softmax_out)
    transfer_loss = loss.CDAN([features, softmax_out], ad_net2, entropy,
                              calc_coeff(i), random_layer=None)
    outputs2 = torch.cat((outputs_source,outputs_source2), dim=0)
    features2 = torch.cat((features_source,features_source2), dim=0)
    labels2 = torch.cat((labels_source,labels_source2), dim=0)
    classifier_loss = nn.CrossEntropyLoss()(outputs2, labels2)
    
    cs_loss_tgt1=nn.CrossEntropyLoss(reduction='none')(outputs_target,fake0)
    cs_loss_tgt2=(cs_loss_tgt1*mask1).mean()
    sfLoss1=softNNLoss(features2,labels2)
    total_loss = 1.0 * transfer_loss + classifier_loss+cs_loss_tgt2+calc_coeff2(i)*sfLoss1
    total_loss.backward()
    optimizer2.step()
    
    ##eval
    base_network.eval()
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for inputs_target, labels_target in LoaderTGte:
            inputs_target=inputs_target.to(device)
            labels_target=labels_target.to(device)
            _, outputs1 = base_network(inputs_target, source=1)
            _, outputs2 = base_network(inputs_target, source=2)
            outputs=outputs1+outputs2
            outputs1 = nn.Softmax(dim=1)(outputs1)
            _, predict1 = torch.max(outputs1, 1)
            correct1 +=torch.sum(predict1==labels_target.data)
            
            outputs2 = nn.Softmax(dim=1)(outputs2)
            _, predict2 = torch.max(outputs2, 1)
            correct2 +=torch.sum(predict2==labels_target.data)
            
            outputs = nn.Softmax(dim=1)(outputs)
            _, predict = torch.max(outputs, 1)
            correct +=torch.sum(predict==labels_target.data)
        AccTe=correct.double()/len(DTsetTGte)
        AccTe1=correct1.double()/len(DTsetTGte)
        AccTe2=correct2.double()/len(DTsetTGte)
        AccTe0=max(AccTe,AccTe1,AccTe2)
    if AccTe0> best_acc:           
        best_acc = AccTe0
        torch.save(base_network, os.path.join(dir_model,'WD2a_P2_{:03d}.pth'.format(i)))
        if best_idx>0:
            os.remove(os.path.join(dir_model,'WD2a_P2_{:03d}.pth'.format(best_idx)))
        best_idx = i
    log_str = "iter: {:05d}, Acc0: {:.5f}, Acc1: {:.5f}, Acc2: {:.5f},trans_loss:{:.4f}, clas_loss:{:.4f}, total_loss:{:.4f}" \
            .format(i,AccTe,AccTe1, AccTe2, transfer_loss.item(), classifier_loss.item(), total_loss.item())
    print(log_str)
print('Best iter is',best_idx,"****best acc ",best_acc)            

    



