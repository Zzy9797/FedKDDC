#main training file
import copy
import math
import random
import time

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import sys
from config import get_args
from utils import aggregation_by_graph, update_graph_matrix_neighbor_distribution,normal_aggregation
from test import compute_acc, compute_local_test_accuracy
from model import simplecnn, textcnn,fashioncnn
from prepare_data import get_dataloader
from attack import *
import torch.nn.functional as F

#local training of FedKDDC, including the training of the personalized models and the global models
def local_train_FedKDDC(args, round, nets_this_round,teachers_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
        
    print('training student')   #train personalized models
    for net_id, net in nets_this_round.items():   #each local training

        teacher=teachers_this_round[net_id]
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:   #first test
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)  

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        kl=nn.KLDivLoss(reduction="batchmean").cuda()

        net.cuda()
        net.train()
        teacher.cuda()
        teacher.eval()
        
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)  
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            with torch.no_grad():
                teacher_out=teacher(x)
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)

            if round>0:
                loss = criterion(out, target)+args.alpha3*kl(F.log_softmax(out, dim=1),F.softmax(teacher_out, dim=1)) #distillation
            else:
                loss = criterion(out,target)
        
                
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:    #second test
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        
        net.to('cpu')
    print('training teachers')   #train global models
    for net_id, teacher in teachers_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

      
        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer_teacher = optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer_teacher = optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer_teacher = optim.SGD(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
       
        teacher.cuda()
        teacher.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl) 
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer_teacher.zero_grad()
            target = target.long()

            out = teacher(x)
            loss = criterion(out, target)
    
                
            loss.backward()
            optimizer_teacher.step()

        teacher.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()

now = int(time.time())   
timeArray = time.localtime(now)
new_time=timeArry
timestr="{}-{}-{}_{}:{}:{}".format(new_time[0],new_time[1],new_time[2],new_time[3],new_time[4],new_time[5]) #timestamp
print(timestr)
args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)


party_list = [i for i in range(args.n_parties)]
party_list_rounds = []

for i in range(args.comm_round):
    party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))  #set benign clients
benign_client_list.sort()
print(f'>> -------- Benign clients: {benign_client_list} --------')

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)   #get data partition
#set models
if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'fashionMnist':
    model=fashioncnn  
elif args.dataset == 'svhn':
    model=simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn       
                  
#initilization of the models    
global_model = model(cfg['classes_size'])
global_parameters = global_model.state_dict()  
global_model_teacher = model(cfg['classes_size'])
global_parameters_teacher = global_model_teacher.state_dict()  
local_models = []
local_models_teacher = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
for i in range(cfg['client_num']):    #initilization of the local models and the accuracy
    local_models.append(model(cfg['classes_size']))
    local_models_teacher.append(model(cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 
graph_matrix[range(len(local_models)), range(len(local_models))] = 0   #initialze aggregation weights

for net in local_models:
    net.load_state_dict(global_parameters)


    
cluster_model_vectors = {}
for round in range(cfg["comm_round"]):
    party_list_this_round = party_list_rounds[round]  #all in
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    teachers_this_round = {k: local_models_teacher[k] for k in party_list_this_round}

    for key,teacher in teachers_this_round.items():
        teachers_this_round[key].load_state_dict(global_parameters_teacher)    #each teacher load global model
    distributions_this_round={k:data_distributions[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
    teachers_param_start = {k: copy.deepcopy(local_models_teacher[k]) for k in party_list_this_round}
    mean_personalized_acc = local_train_FedKDDC(args, round, nets_this_round, teachers_this_round,cluster_model_vectors, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)
   
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)  #attack student model
    manipulate_gradient(args, None, teachers_this_round, benign_client_list, teachers_param_start)  #attack teacher model

    graph_matrix = update_graph_matrix_neighbor_distribution(graph_matrix, nets_this_round, distributions_this_round, global_parameters, dw, fed_avg_freqs, args.alpha1, args.alpha2,args.T,args.difference_measure)   # Graph Matrix is not normalized yet
    cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters)             #aggregate personalized models                                       # Aggregation weight is normalized here

    aggregated_teacher=normal_aggregation(cfg, teachers_this_round, fed_avg_freqs,global_parameters)   #aggregate global models

    global_model_teacher.load_state_dict(aggregated_teacher)
    global_parameters_teacher = global_model_teacher.state_dict()

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)



