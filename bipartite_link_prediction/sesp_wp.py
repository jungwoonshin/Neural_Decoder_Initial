import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp

import numpy as np
import os
import time

from input_data import *
from preprocessing import *
from postprocessing import *
import args
import model
import pickle
from scipy import linalg
from sklearn.utils.extmath import randomized_svd
from sklearn import metrics
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))     

def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]].item())
        # exit()
        if e[0] < len(u2id) and e[1] < len(u2id):
            print('warning1')
        if e[0] > len(u2id) and e[1] > len(u2id):
            print('warning1')

        score = sigmoid(adj_rec[e[0], e[1]].item())
        # print(score)
        # score = adj_rec[e[0], e[1]].item()
        preds.append(score)
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        if e[0] < len(u2id) and e[1] <len(u2id):
            print('warning2')
        if e[0] > len(u2id) and e[1] > len(u2id):
            print('warning2')
        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        score = sigmoid(adj_rec[e[0], e[1]].item())
        # score = adj_rec[e[0], e[1]].item()
        preds_neg.append(score)
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all)
    auc_score = metrics.auc(fpr, tpr)
    # if epoch > 100:
    #     print("ANSWERS")
    #     print(labels_all)
    #     print("PREDICTIONS")
    #     print(preds_all)

    return roc_score, ap_score, auc_score

def getSESP(B_r,B_triangle):
    print('B_r.shape: ',B_r.shape)
    print('B_triangle.shape: ',B_triangle.shape)

    eigVals, eigVecs = linalg.eigh(B_r) 
    eigVals = np.diag(eigVals)
    # val = np.zeros((B_r.shape))
    # for i in range(0, B_r.shape[0]):
    #     val += np.multiply(eigVals[i,i],np.matmul(eigVecs[:,i].reshape(-1,1), eigVecs[:,i].reshape(1,-1)))

    # print('  np.allclose(val, B_r): ',np.allclose(val, B_r, 1e-05,1e-08))

    result = np.zeros((B_r.shape), dtype='float64')
    for i in range(0, B_r.shape[0]):
        left = np.matmul(eigVecs[:,i].reshape(1,-1),B_triangle)
        right = eigVecs[:,i].reshape(-1,1)
        numerator = np.matmul(left,right)
        denominator = np.matmul(eigVecs[:,i].reshape(1,-1),right)
        delta_sigma = numerator/denominator
        delta_sigma = delta_sigma[0][0]

        result += np.multiply((eigVals[i,i] + delta_sigma) ,  np.matmul(eigVecs[:,i].reshape(-1,1), eigVecs[:,i].reshape(1,-1)))

    # B_hat = sigmoid(result)
    B_hat = result
    return result

def getBrAndBtriangle02(adj_train):
    # adj_train += adj_train.T
    # adj_train = adj_train.toarray()

    adj_train = sp.csr_matrix(adj_train)
    # adj_train2[len(u2id):,:len(u2id)] = np.zeros((len(v2id),len(u2id)))
    adj_tuple = sparse_to_tuple(adj_train)
    edges = adj_tuple[0]
    percentage = 0.9
    num_train = int(np.floor(edges.shape[0] * percentage)) # 10%
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    print('num_train: ',num_train)
    B_r_idx = all_edge_idx[:num_train] # 90% of training edges
    # B_r_idx = all_edge_idx # 100% of training edges
    B_triangle_idx = all_edge_idx[num_train:] # 10% of training edges
    B_r_edges = edges[B_r_idx]
    B_triangle_edges = edges[B_triangle_idx]

    print('len(B_r_edges): ', len(B_r_edges))
    print('len(B_triangle_edges): ', len(B_triangle_edges))

    data = np.ones(B_r_edges.shape[0])
    data2 = np.ones(B_triangle_edges.shape[0])

    B_r = sp.csr_matrix((data, (B_r_edges[:, 0], B_r_edges[:, 1])), shape=adj_train.shape).toarray()
    B_r = B_r + B_r.T

    # B_r += sigmoid(np.matmul(adj_train.toarray(),np.transpose(adj_train.toarray())))
    adj_train = adj_train.toarray()
    B = adj_train[:len(u2id), len(u2id):]
    B_T = B.T
    # B_su = sigmoid(B @ B_T)
    B_su = B @ B_T

    B_su = 1 / (1 + np.exp(-B_su))
    # B_sv = sigmoid(B_T @ B)
    B_sv = B_T @ B

    B_sv = 1 / (1 + np.exp(-B_sv))
    B_r[:len(u2id),:len(u2id)] = B_su
    B_r[len(u2id):,len(u2id):] = B_sv

    B_triangle = sp.csr_matrix((data2, (B_triangle_edges[:, 0], B_triangle_edges[:, 1])), shape=adj_train.shape).toarray()
    B_triangle += B_triangle.T

    print('B_r is symmetric: ', check_symmetric(B_r))
    return B_r, B_triangle

test_ap_list = []
test_roc_list = []
test_precision_list = []
for i in range(10):
    adj, features,\
            adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all = get_data(args.dataset)


    with open('data/bipartite/id2name/' +args.dataset +'u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open('data/bipartite/id2name/' +args.dataset +'v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    B_hat = 0
    for i in range(1):
        B_r,B_triangle = getBrAndBtriangle02(adj_train)
        B_hat += getSESP(B_r, B_triangle)
    B_hat/=1.0

    test_precision = get_precision(test_edges, test_edges_false, B_hat, adj_orig, sparse_to_tuple(sparse.csr_matrix(train_edges))[0], u2id, v2id)
    test_roc, test_ap, test_auc = get_scores(test_edges, test_edges_false, B_hat)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap), 
              'test precision=','{:.5f}'.format(test_precision))
    test_roc_list.append(test_roc)
    test_ap_list.append(test_ap)
    test_precision_list.append(test_precision)
    # break

mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))
mean_precision, ste_precision = np.mean(test_precision_list), np.std(test_precision_list)/(args.numexp**(1/2))

print('SESP-WP')

roc = '{:.1f}'.format(mean_roc*100.0)+'+'+'{:.2f}'.format(ste_roc*100.0).strip(' ')
ap = '{:.1f}'.format(mean_ap*100.0)+'+'+'{:.2f}'.format(ste_ap*100.0).strip(' ')
prec = '{:.1f}'.format(mean_precision*100.0)+'+'+'{:.2f}'.format(ste_precision*100.0).strip(' ')

print(roc)
print(ap)
print(prec)
