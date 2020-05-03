import os
import args
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import scipy.sparse as sp
import pickle
import scipy.stats
from collections import defaultdict
import numpy as np
import time

from input_data import *
from preprocessing import *
from postprocessing import *
import model


# Train on CPU (hide GPU) due to memory constraints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reverse_edge(edges):
    reversed_edge = []
    for (i,j) in edges:
        reversed_edge.append((j,i))
    return reversed_edge

def get_train_edges_parts(edges, epoch):
    number_of_slices = int(np.floor(len(edges)/args.subsample_number))
    index = epoch % number_of_slices
    return edges[index*args.subsample_number:(index+1)*args.subsample_number]

def learn_train_adj(seed, model_name):
    global adj_train, adj, features, adj_norm, adj_label, weight_mask, weight_tensor, pos_weight, norm, num_feature, features_nonzero, num_nodes
    global train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, false_edges
    global u2id, v2id
    global adj_orig, adj_unnormalized, adj_norm_first
    global epoch_test_performance, epoch_valid_performance
    # train model

    # init model and optimizer
    if torch.cuda.is_available():
        adj_norm = adj_norm.cuda()

    torch.manual_seed(seed)
    model_adj_norm = getattr(model,model_name)(adj_norm)
    optimizer = Adam(model_adj_norm.parameters(), lr=args.learning_rate)
    # optimizer = SGD(model_adj_norm.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        features = features.cuda()
        adj_label = adj_label.cuda()
        model_adj_norm.cuda()
        weight_tensor = weight_tensor.cuda()


    for epoch in range(args.num_epoch):

        t = time.time() 

        np.random.seed(args.edge_idx_seed_3)
        np.random.shuffle(train_edges)
        np.random.seed(args.edge_idx_seed_3)
        np.random.shuffle(train_false_edges)
        args.edge_idx_seed_3 += 1

        # train_edges_part = get_train_edges_parts(train_edges, epoch)
        # train_false_edges_part = get_train_edges_parts(train_false_edges, epoch)

        train_edges_part = train_edges[:args.subsample_number]
        train_false_edges_part = train_false_edges[:args.subsample_number] 

        # train_edges_part_half = train_edges[:int(args.subsample_number/2)]
        # train_edges_part = np.concatenate((train_edges[:args.subsample_number], reverse_edge(train_edges[:args.subsample_number]))) 
        # train_false_edges_part_half = train_false_edges[:int(args.subsample_number/2)] 
        # train_false_edges_part = np.concatenate((train_false_edges[:args.subsample_number],  reverse_edge(train_false_edges[:args.subsample_number])))

        A_pred = model_adj_norm(features, train_edges_part, train_false_edges_part)
        optimizer.zero_grad()

        loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        if args.model == 'VGAE':
            kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model_adj_norm.logstd - model_adj_norm.mean**2 - torch.exp(model_adj_norm.logstd)).sum(1).mean()
            loss -= kl_divergence

        loss.backward()
        optimizer.step()
        # train_acc = get_acc(A_pred_whole,adj_label)
        with torch.no_grad():
            A_pred = model_adj_norm(features, val_edges, val_edges_false)
        val_roc, val_ap = get_scores_stochastic(val_edges, val_edges_false, A_pred.cpu(),adj_orig)
        # val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred_whole, adj_orig)

        if args.print_val:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                  # "train_acc=", "{:.5f}".format(train_acc),
                  "val_roc=", "{:.5f}".format(val_roc),
                  "val_ap=", "{:.5f}".format(val_ap),
                  "time=", "{:.5f}".format(time.time() - t))
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                A_pred = model_adj_norm(features, test_edges, test_edges_false)
            test_roc, test_ap = get_scores_stochastic(test_edges, test_edges_false, A_pred.cpu(), adj_orig)
            epoch_test_performance[(epoch+1)].append(test_roc)
            epoch_valid_performance[(epoch+1)].append(val_roc)

            print(model_name + " End of training!", "test_roc=", "{:.5f}".format(test_roc),
                        "test_ap=", "{:.5f}".format(test_ap)) 
    with torch.no_grad():
        A_pred = model_adj_norm(features, test_edges, test_edges_false)

    test_roc, test_ap = get_scores_stochastic(test_edges, test_edges_false, A_pred.cpu(), adj_orig)
    print(model_name + " End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap)) 
              # 'test precision=','{:.5f}'.format(test_precision))
    # A_pred = model_adj_norm(features, test_edges, test_edges_false, True)    
    learn_train_adj = A_pred.detach().cpu().numpy()
    learn_train_adj = sp.csr_matrix(learn_train_adj)
    # sp.save_npz('data/'+str(args.model1) +'_' +str(args.dataset)+'_reconstructed_matrix.npz', learn_train_adj)
    # print('feature matrix saved!')
    # exit()
    return learn_train_adj, test_roc, test_ap

def run():

    global adj_train, adj, features, adj_norm, adj_label, weight_mask, weight_tensor, pos_weight, norm, num_feature, features_nonzero, num_nodes
    global train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, adj_all, false_edges
    global u2id, v2id, adj_unnormalized
    global adj_orig, adj_norm_first
    global epoch_test_performance, epoch_valid_performance


    # train model
    test_ap_list = []
    test_roc_list = []
    test_precision_list = []

    test_ap_pretrain_list = []
    test_roc_pretrain_list = []
    test_precision_pretrain_list = []

    test_ap_pretrain_gae_list = []
    test_roc_pretrain_gae_list = []
    test_precision_pretrain_gae_list = []

    epoch_test_performance = defaultdict(list)
    epoch_valid_performance = defaultdict(list)

    for seed in range(args.numexp):

        print('cuda device= '+ str(args.device))
        print('dataset=' + str(args.dataset))
        print('learning rate= '+ str(args.learning_rate))
        print('numexp= '+ str(args.numexp))
        print('epoch= '+ str(args.num_epoch))
        print('subsample_number='+str(args.subsample_number))
        print('hidden_dim1='+str(args.hidden1_dim))
        
        adj, features,\
            adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all = get_data(args.dataset)

        with open('data/bipartite/id2name/'+ str(args.dataset) +'u2id.pkl', 'rb') as f:
            u2id = pickle.load(f)
        with open('data/bipartite/id2name/'+ str(args.dataset) +'v2id.pkl', 'rb') as f:
            v2id = pickle.load(f)

        if args.subsample_number > len(train_edges):
            args.subsample_number = len(train_edges)
            # args.subsample_number = len(train_edges) if len(train_edges) % 2 == 0 else len(train_edges) -1


        adj_orig = adj  
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train = adj_train + adj_train.T
        # adj_train_2 = adj_train[:len(u2id),len(u2id):].copy()

        # adj = adj_train_2
        adj = adj_train

        num_nodes = adj.shape[0]

        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                    torch.FloatTensor(features[1]), 
                                    torch.Size(features[2]))
        
        # Some preprocessing
        adj_norm = preprocess_graph_neg_one(adj)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                    torch.FloatTensor(adj_norm[1]), 
                                    torch.Size(adj_norm[2]))
        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


        adj_label = adj_train + sp.eye(adj_train.shape[0])
        # adj_label = adj_train_2
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                    torch.FloatTensor(adj_label[1]), 
                                    torch.Size(adj_label[2]))
        adj_label = torch.cat((torch.ones(args.subsample_number,1), torch.zeros(args.subsample_number,1)),0)

        weight_mask = adj_label.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)) 
        weight_tensor[weight_mask] = pos_weight

        adj_unnormalized = sp.coo_matrix(adj)
        adj_unnormalized = sparse_to_tuple(adj_unnormalized)
        adj_unnormalized = torch.sparse.FloatTensor(torch.LongTensor(adj_unnormalized[0].T), 
                                    torch.FloatTensor(adj_unnormalized[1]), 
                                    torch.Size(adj_unnormalized[2]))


        print('='*88)
        print(str(seed)+' iteration....' + str(args.learning_rate))
        print('='*88)

        adj_train_norm, test_roc_pretrain, test_ap_pretrain = learn_train_adj(seed, 'ND_LGAE')

        test_ap_pretrain_list.append(test_ap_pretrain)
        test_roc_pretrain_list.append(test_roc_pretrain)



    mean_roc_pretrain, ste_roc_pretrain = np.mean(test_roc_pretrain_list), np.std(test_roc_pretrain_list)/(args.numexp**(1/2))
    mean_ap_pretrain, ste_ap_pretrain = np.mean(test_ap_pretrain_list), np.std(test_ap_pretrain_list)/(args.numexp**(1/2))

    print('cuda device= '+ str(args.device))
    print('dataset=' + str(args.dataset))
    print('learning rate= '+ str(args.learning_rate))
    print('numexp= '+ str(args.numexp))
    print('epoch= '+ str(args.num_epoch))
    print('subsample_number='+str(args.subsample_number))
    print('hidden_dim1='+str(args.hidden1_dim))

    roc = '{:.1f}'.format(mean_roc_pretrain*100.0)+'+'+'{:.2f}'.format(ste_roc_pretrain*100.0).strip(' ')
    ap = '{:.1f}'.format(mean_ap_pretrain*100.0)+'+'+'{:.2f}'.format(ste_ap_pretrain*100.0).strip(' ')
    
    print('Neural Decoder')
    print(roc)
    print(ap)

    epoch_test_performance_list = []
    for key,val in epoch_test_performance.items():
        # print(key + ':' + np.mean(val))
        epoch_test_performance_list.append((key, np.mean(val)))

    sorted_by_second = sorted(epoch_test_performance_list, key=lambda tup: tup[1],reverse=True)
    print('test=',sorted_by_second)

    epoch_valid_performance_list = []
    for key,val in epoch_valid_performance.items():
        # print(key + ':' + np.mean(val))
        epoch_valid_performance_list.append((key, np.mean(val)))

    sorted_by_second = sorted(epoch_valid_performance_list, key=lambda tup: tup[1],reverse=True)
    print('valid=',sorted_by_second)


run()
