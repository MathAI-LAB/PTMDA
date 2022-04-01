#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:40:57 2020

@author: user9
"""

##############################################
import os
os.chdir('/data2/yonghui_/Bn_Spur1')
import pandas as pd
data0=pd.read_csv('../data1_/office/amazon_list.txt',sep=" ",header=None)
p1=pd.read_csv('./Pred1/WD2a3_Rs50_8721.txt',sep=" ",header=None)
p1[3]=0
p1.loc[p1.iloc[:,0]>9.99999,3]=1
# p1.iloc[:,3].sum()

ps5=p1.loc[p1.iloc[:,3]==1,:]
((ps5.iloc[:,1])==ps5[2]).sum()/ps5.shape[0]
((p1.iloc[:,1])==p1[2]).sum()/p1.shape[0]
ot1=data0
ot1[1]=p1.iloc[:,1]
ot1[2]=p1.iloc[:,3]
# ot1.iloc[:,2].sum()
ot1.to_csv("Aprd1.txt",sep=" ",columns=None,index=False,header=False)