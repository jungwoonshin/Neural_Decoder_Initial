import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, confusion_matrix
import scipy.stats
from scipy import sparse
import networkx as nx
from scipy.stats import pearsonr

from input_data import *
from preprocessing import *
from postprocessing import *
import pickle
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))     
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_precision_scores(edges_pos, edges_neg, adj_rec, adj_orig, adj_norm):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    precision, recall, thresholds = precision_recall_curve(adj_orig.flatten(), adj_rec.flatten())
    with np.errstate(divide='ignore'):
        f1 = 2/(1/precision+1/recall)
    # f1 = precision+recall
    max_index = np.argmax(f1)
    max_threshold = thresholds[max_index]
    # print(max_threshold)

    adj_rec[adj_rec > max_threshold] = 1.0
    adj_rec[adj_rec <= max_threshold] = 0.0

        # Predict on test set of edges
    preds = []
    pos = []


    predicted_to_be_false_but_true =[]
    for e in edges_pos:
        if (adj_rec[e[0], e[1]].item() == 0.0):
            predicted_to_be_false_but_true.append((e[0],e[1]))

        # preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        preds.append(adj_rec[e[0], e[1]].item())    
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(adj_rec[e[0], e[1]].item())
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    cm = confusion_matrix(labels_all, preds_all,labels=[1,0])
    print(cm)


    return f1[max_index],max_threshold, predicted_to_be_false_but_true

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]].item())
        # preds.append(adj_rec[e[0], e[1]].item())

        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]].item())
        # preds_neg.append(adj_rec[e[0], e[1]].item())

        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


def get_scores_stochastic(edges_pos, edges_neg, adj_rec,adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for index, e in enumerate(edges_pos):
        preds.append(sigmoid(adj_rec[index,:].item()))
        # preds.append(adj_rec[e[0], e[1]].item())

        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for index, e in enumerate(edges_neg):
        index += len(edges_pos)
        preds_neg.append(sigmoid(adj_rec[index,:].item()))
        # preds_neg.append(adj_rec[e[0], e[1]].item())

        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_correlation(edges_pos, edges_neg, adj_rec, adj_orig):


    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].item()))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    corr, _ = pearsonr(preds_all, labels_all)
    print('correlation:' ,corr)

    return corr

def get_precision(edges_pos, edges_neg, adj_rec, adj_orig,train_edges, u2id,v2id):
    train_edges_list = []
    for i in train_edges:
        train_edges_list.append((i[0],i[1]))

    all_edges = []
    for i in range(len(u2id)):
        for j in range(len(u2id),len(u2id)+len(v2id)):
            all_edges.append((i,j))
    # all_edges = [(x,y) for x in range(adj_rec.shape[0]) for y in range(adj_rec.shape[0])]

    equal_edges =  [(x,x) for x in range(adj_rec.shape[0])]
    u_minus_ep = list(set(all_edges) - set(equal_edges))
    u_minus_ep = list(set(u_minus_ep) - set(train_edges_list))

    index_value_list = []
    for i in u_minus_ep:
        x = i[0]
        y = i[1]
        element = (x,y, adj_rec[x,y].item())
        index_value_list.append(element)

    index_value_list = sorted(index_value_list, key=lambda tup: tup[2], reverse=True)
    index_value_list = index_value_list[:len(edges_pos)]
    correct = 0.
    for (x,y,z) in index_value_list:
        if (x,y) in edges_pos:
            correct+=1.
    precision = correct / len(edges_pos)
    # print(precision)
    return precision

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def max1(t1, t2):
    combined = torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2)
    return torch.max(combined, dim=2)[0].squeeze(2)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_jc_scores(adj_train,u2id,v2id):
    adj_train_np = adj_train.toarray()
    graph =nx.from_numpy_matrix(adj_train.toarray())
    adj_train = nx.to_dict_of_lists(graph)

    # Predict on test set of edges
    S = np.zeros(adj_train_np.shape)
    a = adj_train_np.shape[0]
    b1 = 0
    b2 = adj_train_np.shape[0]
    for x in range(a):
        for y in range(b1,b2):
            idx =  adj_train[x]
            Sx = [adj_train[x] for x in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[y]
            Sy = set(Sy)
            score = 0.0
            intersection1 = Sx.intersection(Sy)

            idx =  adj_train[y]
            Sx = [adj_train[y] for y in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[x]
            Sy = set(Sy)
            score = 0.0
            intersection2 = Sx.intersection(Sy)
            numerator = intersection1.union(intersection2)

            Sx = set(adj_train[x])
            Sy = set(adj_train[y])
            denominator = Sx.union(Sy)

            score = len(numerator)/len(denominator) if len(Sx.union(Sy))>0 else 0.0
            S[x,y] = score
            S[y,x] = score
        b1+=1
    return S

def get_cn_scores(adj_train,u2id,v2id):
    adj_train_np = adj_train.toarray()
    graph =nx.from_numpy_matrix(adj_train.toarray())
    adj_train = nx.to_dict_of_lists(graph)

    # Predict on test set of edges
    S = np.zeros(adj_train_np.shape)
    a = adj_train_np.shape[0]
    b1 = 0
    b2 = adj_train_np.shape[0]
    for x in range(a):
        for y in range(b1,b2):
            idx =  adj_train[x]
            Sx = [adj_train[x] for x in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[y]
            Sy = set(Sy)
            score = 0.0
            intersection1 = Sx.intersection(Sy)

            idx =  adj_train[y]
            Sx = [adj_train[y] for y in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[x]
            Sy = set(Sy)
            score = 0.0
            intersection2 = Sx.intersection(Sy)
            numerator = intersection1.union(intersection2)

            Sx = set(adj_train[x])
            Sy = set(adj_train[y])
            denominator = Sx.union(Sy)

            score = len(numerator)
            S[x,y] = score
            S[y,x] = score
        b1+=1
    return S
    
def get_aa_scores(adj_train,u2id,v2id):
    adj_train_np = adj_train.toarray()
    graph =nx.from_numpy_matrix(adj_train.toarray())
    adj_train = nx.to_dict_of_lists(graph)

    # Predict on test set of edges
    S = np.zeros(adj_train_np.shape)
    a = adj_train_np.shape[0]
    b1 = 0
    b2 = adj_train_np.shape[0]
    for x in range(a):
        for y in range(b1,b2):
            idx =  adj_train[x]
            Sx = [adj_train[x] for x in idx]
            Sx = set([item for sublist in Sx for item in sublist])

            Sy = adj_train[y]
            Sy = set(Sy)
            intersection1 = Sx.intersection(Sy)

            idx =  adj_train[y]

            Sx = [adj_train[y] for y in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[x]
            Sy = set(Sy)
            intersection2 = Sx.intersection(Sy)
            intersection = intersection1.union(intersection2)
            score=0.0
            for i in intersection:
                try:
                    if len(adj_train[i]) == 1.0:
                        neighbor_z = 1./math.log(len(adj_train[i])+0.1)
                    else:
                        neighbor_z = 1./math.log(len(adj_train[i]))
                except ZeroDivisionError:
                    #neighbor_z = 1/math.log(0.0000001)
                    neighbor_z = 0.
                    # print('zerodivision')
                score += neighbor_z
            S[x,y] = score
            S[y,x] = score
        b1+=1
    return S

def get_cpa_scores(adj_train,u2id,v2id):
    adj_train_np = adj_train.toarray()
    graph =nx.from_numpy_matrix(adj_train.toarray())
    adj_train = nx.to_dict_of_lists(graph)

    # Predict on test set of edges
    S = np.zeros(adj_train_np.shape)
    a = adj_train_np.shape[0]
    b1 = 0
    b2 = adj_train_np.shape[0]
    for x in range(a):
        for y in range(b1,b2):
            idx =  adj_train[x]

            Sx = [adj_train[x] for x in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[y]
            Sy = set(Sy)
            score = 0.0
            intersection1 = Sx.intersection(Sy)

            idx =  adj_train[y]

            Sx = [adj_train[y] for y in idx]
            Sx = set([item for sublist in Sx for item in sublist])
            # print(Sx)
            Sy = adj_train[x]
            Sy = set(Sy)
            score = 0.0
            intersection2 = Sx.intersection(Sy)
            numerator = intersection1.union(intersection2)

            Sx = adj_train[x]
            Sy = adj_train[y]

            # print(adj_train_numpy_matrix[244,0])
            lcl = 0.
            for a in Sx:
                for b in Sy:
                    if adj_train_np[a,b] == 1.0 or adj_train_np[b,a] == 1.0:
                        lcl+=1.

            car = len(numerator) * lcl
            e_x = len(adj_train[x])
            e_y = len(adj_train[y])
            score = (e_x * e_y) + (e_x *car) + (e_y * car) + (car**2.0)
            S[x,y] = score
            S[y,x] = score
        b1+=1
    return S