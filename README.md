# FedKDDC
Knowledge-Distillation based Personalized Federated Learning with Distribution Constraints

## Experiments
Our experiments are implemented with open-source PyTorch 1.13.1, on an NVIDIA GeForce RTX 4090 platform. 
The results are shown in Tables 1-7.

### Table 1: 
Accuracy (%) comparisons of image classification task on CIFAR-10 and CIFAR-100 under data partitioning of Dirichlet distribution. 

 Method  | CIFAR-10  | CIFAR-100
 ---- | ----- | ------  
 FedAvg  | 62.92 | 27.78 
 FedAvg-FT  | 84.09 | 50.58 
 FedProx  | 62.25 | 27.87 
 FedProx-FT  | 83.71 | 50.99 
 CFL  | 83.84 | 49.12 
 Per-FedAvg  | 84.02 | 50.38
 pFedMe  | 75.24 | 34.37
 FedAMP  | 75.49 | 31.04
 Ditto  | 83.78 | 50.33
 FedRep  | 83.47 | 50.15
 pFedHN  | 82.57 | 49.08
 FedRep  | 83.47 | 50.15
 FedRoD  | 83.49 | 47.96
 kNN-Per  | 70.05 | 25.84
 pFedGraph  | 84.28 | 51.63
 FedKDDC  | 85.69±0.04 | 54.33±0.22

### Table 2:
Accuracy (%) comparisons of image classification task on SVHN and Fashion-MNIST under data partitioning of Dirichlet distribution. 

| Method | SVHN | Fashion-MNIST |
| --- | --- | --- |
| FedAvg | 85.86 | 84.51 |
| FedAvg-FT | 90.11 | 96.33 |
| FedProx | 85.98 | 82.79 |
| FedProx-FT | 89.26 | 96.27 |
| CFL | 89.53 | 96.64 |
| Per-FedAvg | 89.82 | 96.30 |
| pFedMe | 82.59 | 93.34 |
| FedAMP | 67.62 | 93.13 |
| Ditto | 90.13 | 95.97 |
| FedRep | 89.32 | 96.44 |
| pFedHN | 78.44 | 95.87 |
| FedRoD | 89.13 | 96.46 |
| kNN-Per | 86.21 | 91.87 |
| pFedGraph | 89.59 | 96.46 |
| FedKDDC | 90.66±0.15 | 96.57±0.05 |

### Table 3:
Accuracy (%) comparisons of text classification task on the dataset Yahoo! Answers under data partitioning of Dirichlet distribution.
Method | Accuracy | Method | Accuracy 
| --- | --- | --- | --- |
FedAvg | 51.99 | FedAMP | 64.26
FedAvg-FT | 80.24 | Ditto | 79.10
FedProx | 50.82 | FedRep | 76.29
FedProx-FT | 79.84 | FedRoD | 78.82
CFL | 79.40 | pFedGraph | 80.56
Per-FedAvg | 80.16 | FedKDDC | 85.50±0.09
pFedMe | 58.91 | | 

### Table 4:
Accuracy (%) comparisons of image classification task on CIFAR-10 and CIFAR-100 under data partitioning of pathological distribution.

 Method  | CIFAR-10  | CIFAR-100
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
| FedKDDC | 92.84±0.09 | 58.82±0.12

### Table 5:
Accuracy (%) comparisons of text classification task on the dataset Yahoo! Answers under data partitioning of pathological distribution.


Method | Accuracy | Method | Accuracy 
| --- | --- | --- | --- |
FedAvg | 63.14 | FedAMP | 76.85 
FedAvg-FT | 87.02 | Ditto | 86.33 
FedProx | 49.30 | FedRep | 86.00 
FedProx-FT | 86.72 | FedRoD | 86.51 
CFL | 88.61 | pFedGraph | 89.04 
Per-FedAvg | 86.92 | FedKDDC | 92.16±0.15 
pFedMe | 64.16 | | 

### Table 6:
Accuracy (%) comparisons of image classification task on the datasets CIFAR-10 and CIFAR-100 under homogeneous data partitioning.

 Method  | CIFAR-10  | CIFAR-100
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
FedKDDC | 67.55±0.09 | 33.65±0.10 

### Table 7:
Accuracy comparisons on CIFAR-10 under poisoning attacks. The data partition follows the Dirichlet distribution. The attack ratio is 0.2.


Method | Accuracy | Method | Accuracy 
| --- | --- | --- | --- |
FedAvg | 39.29  | FedAMP | 75.31 
FedAvg-FT | 67.85 | Ditto | 58.38 
FedProx | 39.72 | FedRep | 68.00 
FedProx-FT | 67.18 | FedRoD | 68.58 
CFL | 67.45 | pFedGraph | 82.39 
Per-FedAvg | 67.57 | FedKDDCm | 82.83 
pFedMe | 64.45 | | 
