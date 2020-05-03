import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

from input_data import load_data
from preprocessing import *
import args
import model

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

auc_list = []
ap_list = []
for _ in range(10):


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


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
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
            # preds.append(sigmoid(adj_rec[index,:].item()))
            preds.append(adj_rec[e[0], e[1]].item())

            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for index, e in enumerate(edges_neg):
            index += len(edges_pos)
            # preds_neg.append(sigmoid(adj_rec[index,:].item()))
            preds_neg.append(adj_rec[e[0], e[1]].item())

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

    # train model
    for epoch in range(args.num_epoch):
        t = time.time()
        np.random.shuffle(train_edges)
        np.random.shuffle(train_false_edges)

        train_edges_part = train_edges[:args.subsample_number]
        train_false_edges_part = train_false_edges[:args.subsample_number]

        A_pred = model_adj_norm(features, train_edges_part, train_false_edges_part)
        optimizer.zero_grad()


        # loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)

        if args.model == 'VGAE' or args.model == 'VGAE2':
            kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model_adj_norm.logstd - model_adj_norm.mean**2 - torch.exp(model_adj_norm.logstd)).sum(1).mean()
            loss -= kl_divergence

        loss.backward()
        optimizer.step()

        train_acc = get_acc(A_pred,adj_label.to_dense())

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred.cpu())
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))

    auc_list.append(test_roc)
    ap_list.append(test_ap)


mean_roc, ste_roc = np.mean(auc_list), np.std(auc_list)/(10**(1/2))
mean_ap, ste_ap = np.mean(ap_list), np.std(ap_list)/(10**(1/2))

roc = '{:.1f}'.format(mean_roc*100.0)+'+'+'{:.2f}'.format(ste_roc*100.0).strip(' ')
ap = '{:.1f}'.format(mean_ap*100.0)+'+'+'{:.2f}'.format(ste_ap*100.0).strip(' ')

print('GAE')
print(roc)
print(ap)
