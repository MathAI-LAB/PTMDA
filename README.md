# PTMDA
This is the pytorch demo code for Multi-Source Unsupervised Domain Adaptation via Pseudo Target Domain, (PTMDA) (IEEE Transactions on Image Processing, 2022). 


## Requirements:

python=3.6.9
pytorch== 1.1.0
torchvision ==0.5.0
numpy==1.19.5
pip==21.0.1
wheel==0.36.2
cuda==9.0


## Files structures:

├─data1_/
│  ├─webcam/
│  │  ├─images/
│  ├─amazon/
│  │  ├─images/


├─(Current directory)
│  PTMDAtrain1.py
│  funCDAN2.py
│  network3.py
│  pre_process.py
│  data_list2.py
│  BNs9.py
│  loss.py
│  logs1.py
   │  data1 (directory of data files)


Run:
step1, run Train1_S1.py.
step2, run Fea_WD2a2.py and sum3.py,
then,  run Train2_S2.py .
