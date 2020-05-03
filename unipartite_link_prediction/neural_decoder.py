import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import time
from collections import defaultdict

from input_data import load_data
from preprocessing import *
import args
import model


# Train on CPU (hide GPU) due to memory constraints
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

auc_list = []
ap_list = []

epoch_test_performance = defaultdict(list)
epoch_valid_performance = defaultdict(list)

def get_train_edges_parts(edges, epoch):
    number_of_slices = int(np.floor(len(edges)/args.subsample_number))
    index = epoch % number_of_slices
    return edges[index*args.subsample_number:(index+1)*args.subsample_number]

for exp in range(10):
    args.model = 'NLGF'

    print('model= '+ str(args.model))     
    print('dataset=' + str(args.dataset))
    print('learning rate= '+ str(args.learning_rate))
    print('epoch= '+ str(args.num_epoch))
    print('subsample_number='+ str(args.subsample_number))
    print('hidden1_dim='+str(args.hidden1_dim))

    adj, features = load_data(args.dataset)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))
    adj_label = torch.cat((torch.ones(args.subsample_number,1), torch.zeros(args.subsample_number,1)),0)

    weight_mask = adj_label.view(-1) == 1
    # weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight


    if torch.cuda.is_available():
        adj_norm = adj_norm.cuda()
    # init model and optimizer
    model_adj_norm = getattr(model,args.model)(adj_norm)
    optimizer = Adam(model_adj_norm.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        features = features.cuda()
        adj_label = adj_label.cuda()
        model_adj_norm.cuda()
        weight_tensor = weight_tensor.cuda()

    def get_scores(edges_pos, edges_neg, adj_rec):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for index, e in enumerate(edges_pos):
            preds.append(adj_rec[index,:].item())
            # preds.append(adj_rec[e[0], e[1]].item())

            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for index, e in enumerate(edges_neg):
            index += len(edges_pos)
            preds_neg.append(adj_rec[index,:].item())
            # preds_neg.append(adj_rec[e[0], e[1]].item())

            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    def reverse_edge(edges):
        reversed_edges = []
        for x,y in edges:
            reversed_edges.append((y,x))
        return reversed_edges
    print('='*88)
    print(str(exp)+' iteration....' + str(args.learning_rate))
    print('='*88)
    
    all_time =0
    # train model
    for epoch in range(args.num_epoch):

        t = time.time()

        np.random.seed(args.edge_idx_seed)
        np.random.shuffle(train_edges)
        np.random.seed(args.edge_idx_seed)
        np.random.shuffle(train_false_edges)

        train_edges_part = train_edges[:args.subsample_number]
        train_false_edges_part = train_false_edges[:args.subsample_number]

        # train_edges_part = get_train_edges_parts(train_edges, epoch)
        # train_false_edges_part = get_train_edges_parts(train_false_edges, epoch)

        # train_edges_part = np.concatenate((get_train_edges_parts(train_edges, epoch), reverse_edge(get_train_edges_parts(train_edges, epoch))))
        # train_false_edges_part = np.concatenate((get_train_edges_parts(train_false_edges, epoch), reverse_edge(get_train_edges_parts(train_false_edges, epoch))))

        model_adj_norm.train()


        A_pred = model_adj_norm(features, train_edges_part, train_false_edges_part)
        optimizer.zero_grad()

        loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        if args.model == 'VGAE' or args.model == 'VGAE2' or args.model =='NVGF':
            kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model_adj_norm.layer.logstd - model_adj_norm.layer.mean**2 - torch.exp(model_adj_norm.layer.logstd)).sum(1).mean()
            loss -= kl_divergence

        loss.backward()
        optimizer.step()

        train_acc = get_acc(A_pred,adj_label)
        time_elapsed = time.time() - t
        all_time += time_elapsed
        with torch.no_grad():
            A_pred = model_adj_norm(features, val_edges, val_edges_false)
        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time_elapsed))
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                A_pred = model_adj_norm(features, test_edges, test_edges_false)
            test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
            epoch_test_performance[(epoch+1)].append(test_roc)
            epoch_valid_performance[(epoch+1)].append(val_roc)

            print(args.model + " End of training!", "test_roc=", "{:.5f}".format(test_roc),
                        "test_ap=", "{:.5f}".format(test_ap)) 

    with torch.no_grad():
        A_pred = model_adj_norm(features, test_edges, test_edges_false)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))

    auc_list.append(test_roc)
    ap_list.append(test_ap)



mean_roc, ste_roc = np.mean(auc_list), np.std(auc_list)/(10**(1/2))
mean_ap, ste_ap = np.mean(ap_list), np.std(ap_list)/(10**(1/2))

roc = '{:.1f}'.format(mean_roc*100.0)+'+'+'{:.2f}'.format(ste_roc*100.0).strip(' ')
ap = '{:.1f}'.format(mean_ap*100.0)+'+'+'{:.2f}'.format(ste_ap*100.0).strip(' ')

print('model= '+ str(args.model))     
print('dataset=' + str(args.dataset))
print('learning rate= '+ str(args.learning_rate))
print('epoch= '+ str(args.num_epoch))
print('subsample_number='+ str(args.subsample_number))
print('hidden1_dim='+str(args.hidden1_dim))

print('\nGAE')
print(roc)
print(ap)
print()

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
