# FedKDDC
Knowledge-Distillation based Personalized Federated Learning with Distribution Constraints

## Code Descriptions
- run.sh: quick strat
- train.py: main code of the algorithm
- utils.py: some utilities of the algorithm
- test.py: evalution indicators
- prepare_data.py、data_partition、cvdataset.py、nlpdataset.py：get loaders
- attacks.py: attack functions
- config.py: configurations
- model.py: backbones
  
## Experiments
Our experiments are implemented with open-source PyTorch 1.13.1, on an NVIDIA GeForce RTX 4090 platform. 

**Results of Dirichlet Distribution**: 
Dirichlet distribution is a typical data splitting principle in FL, which effectively mimics the heterogeneity of data in real applications. We conduct the experiments under Dirichlet distribution data partition to compare the performance among the state-of-the-arts methods.

Table 1: Accuracy (%) comparisons of image classification task on CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST under data partitioning of Dirichlet distribution. 

 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">SVHN&emsp;&emsp;&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">Fashion-MNIST&emsp;&emsp;&emsp;&emsp;&emsp;</span>
  --- | --- | --- | ---| ---
 FedAvg  | 62.92 | 27.78 | 85.86 | 84.51
 FedAvg-FT  | 84.09 | 50.58 | 90.11 | 96.33
 FedProx  | 62.25 | 27.87 | 85.98 | 82.79
 FedProx-FT  | 83.71 | 50.99 | 89.26 | 96.27 
 CFL  | 83.84 | 49.12 | 89.53 | 96.64
 Per-FedAvg  | 84.02 | 50.38 | 89.82 | 96.30
 pFedMe  | 75.24 | 34.37 | 82.59 | 93.34
 FedAMP  | 75.49 | 31.04 | 67.62 | 93.13
 Ditto  | 83.78 | 50.33 | 90.13 | 95.97
 FedRep  | 83.47 | 50.15 | 89.32 | 96.44
 pFedHN  | 82.57 | 49.08 | 78.44 | 95.87
 FedRoD  | 83.49 | 47.96 | 89.13 | 96.46
 kNN-Per  | 70.05 | 25.84 | 96.21 | 91.87
 pFedGraph  | 84.28 | 51.63 | 89.59 | 96.46
 FedKDDC  | **85.69**±0.04 | **54.33**±0.22 | **90.66**±0.15 | **96.57**±0.05


Table 2: Accuracy (%) comparisons of text classification task on the dataset Yahoo! Answers under data partitioning of Dirichlet distribution.
<span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | <span style="white-space:nowrap;">Accuracy&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | <span style="white-space:nowrap;">Accuracy&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>
| --- | --- | --- | --- |
FedAvg | 51.99 | FedAMP | 64.26
FedAvg-FT | 80.24 | Ditto | 79.10
FedProx | 50.82 | FedRep | 76.29
FedProx-FT | 79.84 | FedRoD | 78.82
CFL | 79.40 | pFedGraph | 80.56
Per-FedAvg | 80.16 | FedKDDC | **85.50**±0.09
pFedMe | 58.91 | | 

**Results of Pathological Distribution**:
Consider the scenario where some categories of data are missing on the clients, we also split the datasets based on pathological distribution and conduct the experiments.

Table 3: Accuracy (%) comparisons of image classification task on CIFAR-10 and CIFAR-100 under data partitioning of pathological distribution.
 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>  
| ---- | ----- | ------  
| FedAvg | 66.19 | 26.23 
| FedAvg-FT | 90.20 | 51.12 
| FedProx | 55.76 | 25.64 
| FedProx-FT | 90.37 | 49.91 
| CFL | 90.76 | 52.43 
| Per-FedAvg | 89.60 | 51.14 
| pFedMe | 81.73 | 33.48 
| FedAMP |  86.90 | 37.50 
| Ditto | 89.41 | 50.54 
| FedRep | 90.02 | 51.72 
| pFedHN | 89.91 | 49.06 
| FedRoD | 90.66 |  49.91 
| kNN-Per | 79.09 | 24.70 
| pFedGraph | 92.74 | 56.79 
| FedKDDC | **92.84**±0.09 | **58.82**±0.12

Table 4: Accuracy (%) comparisons of text classification task on the dataset Yahoo! Answers under data partitioning of pathological distribution.
<span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | <span style="white-space:nowrap;">Accuracy&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | <span style="white-space:nowrap;">Accuracy&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>
| --- | --- | --- | --- |
FedAvg | 63.14 | FedAMP | 76.85 
FedAvg-FT | 87.02 | Ditto | 86.33 
FedProx | 49.30 | FedRep | 86.00 
FedProx-FT | 86.72 | FedRoD | 86.51 
CFL | 88.61 | pFedGraph | 89.04 
Per-FedAvg | 86.92 | FedKDDC | **92.16**±0.15 
pFedMe | 64.16 | | 

**Results of Homogeneous Data Partition**:
When data is homogeneous, excessive personalization can sometimes affect model performance. To validate the effectiveness of proposed FedKDDC under data homogeneity, we conduct the experiments on CIFAR-10 and CIFAR-100 under independent and identically distributed data partition.

Table 5: Accuracy (%) comparisons of image classification task on the datasets CIFAR-10 and CIFAR-100 under homogeneous data partitioning.
 <span style="white-space:nowrap;">Method&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> |<span style="white-space:nowrap;">CIFAR-10&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>  |<span style="white-space:nowrap;">CIFAR-100&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span>  
 ---- | ----- | ------  
FedAvg | 67.12 | 31.10 
FedAvg-FT | 63.09 | 25.47 
FedProx | 67.07 | 30.55 
FedProx-FT | 61.93 | 25.15 
CFL | 60.55 | 19.31 
Per-FedAvg | 63.24 | 25.18 
pFedMe | 47.48 | 13.18 
FedAMP |  45.49 | 10.07 
Ditto | 65.35 | 29.41 
FedRep | 62.88 | 21.53 
pFedHN | 62.78 | 25.94 
FedRoD | 62.07 |  18.71 
kNN-Per | 67.01 | 31.04 
pFedGraph | 67.37 | 31.16 
FedKDDC | **67.55**±0.09 | **33.65**±0.10 

## Acknowledgments
Our work is based on the following work, thanks for the code:

https://github.com/MediaBrain-SJTU/pFedGraph

