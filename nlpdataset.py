import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_partition import partition_data
import torch
import os
from torchtext.datasets import YahooAnswers
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    texts = [text for text,label in batch]
    labels= [label for text,label in batch]
    padded_texts = pad_sequence(texts, batch_first=False,padding_value=0)
    padded_texts=padded_texts.permute(1,0)
    labels= torch.tensor(labels)
    
    return padded_texts,labels
class SentDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, i):
        return torch.LongTensor(self.data[i]), self.labels[i]


    def __len__(self) -> int:
        return self.labels.shape[0]
    

def nlpdataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    if dataset == "yahoo_answers":
        traindata = torch.load('./data/yahoo_answers/id/train.pth')
        testdata = torch.load('./data/yahoo_answers/id/test.pth')
        train_data = np.array(traindata['sentences'])
        train_label = traindata['labels']
        train_label=[int(i)-1 for i in train_label]
        train_label=np.array(train_label)
        test_data = np.array(testdata['sentences'])
        test_label = testdata['labels']
        test_label=[int(i)-1 for i in test_label]
        test_label=np.array(test_label)
        n_train = train_label.shape[0]
        h = np.zeros(10)
        for i in test_label:
            h[i]+=1
        print(h)
            
        net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
        train_dataloaders = []
        val_dataloaders = []
        for i in range(n_parties):
            train_idxs = net_dataidx_map[i][:int(0.8*len(net_dataidx_map[i]))]
            val_idxs = net_dataidx_map[i][int(0.8*len(net_dataidx_map[i])):]
            train_dataset = SentDataset(data=train_data[train_idxs], labels=train_label[train_idxs])
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate_fn)
            val_dataset = SentDataset(data=train_data[val_idxs], labels=train_label[val_idxs])
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)
            train_dataloaders.append(train_loader)
            val_dataloaders.append(val_loader)
    test_dataset = SentDataset(data=test_data, labels=test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,collate_fn=custom_collate_fn)
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions