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

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA, NMF, non_negative_factorization
import nimfa
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


#####################################
#   select one model to run SRNMF   #
#   then run srnmf.py               #
#####################################
args.model = 'srnmf_cn'
# args.model = 'srnmf_jc'
# args.model = 'srnmf_cpa'


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def getXY(S, adj_train, k):
    model = NMF(n_components=k, init='nndsvda',solver='mu',max_iter=400)
    X = model.fit_transform(adj_train.toarray())
    Y = model.components_
    print('finished computing factorizing XY')

    # print('X.shape:',X.shape)
    # print('Y.shape:',Y.shape)
    adj_train = adj_train.toarray()

    gamma = 0.5
    lamda = 2.0
    for i in range(200):
        new_X_num = adj_train@Y.T+ gamma*(S*adj_train) @ Y.T
        new_X_den = X@Y@Y.T + gamma * (S * (X@Y)) @ Y.T + lamda * X 
        # new_X_den = np.where(new_X_den != 0, new_X_den, 0.0001 )
        new_X_den = np.where(new_X_den==0, 0.0001, new_X_den) 
        new_X = new_X_num / new_X_den 

        new_Y_num = X.T@adj_train + gamma * X.T @(S*adj_train)
        new_Y_den = X.T@X@Y+gamma*X.T@(S*(X@Y))+lamda*Y
        # new_Y_den = np.where(new_Y_den != 0, new_Y_den, 0.0001 )
        new_Y_den = np.where(new_Y_den==0, 0.0001, new_Y_den) 
        new_Y = new_Y_num / new_Y_den 

        X_old = X
        Y_old = Y

        X = np.multiply(X,new_X)
        Y = np.multiply(Y,new_Y)

    #     diff1 = np.linalg.norm(X_old-X, 'fro')
    #     diff2 = np.linalg.norm(Y_old-Y, 'fro')
    #     print(diff1)
    #     print(diff2)
    # exit()
    return X,Y

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
    adj_train += adj_train.T
    
    pca = PCA().fit(adj_train.toarray())
    cumulative_contribution_rate = list(np.cumsum(pca.explained_variance_ratio_))
    val = min(cumulative_contribution_rate, key=lambda x:abs(x-0.95))
    k = cumulative_contribution_rate.index(val)+1
    print(k)

    if args.model == 'srnmf_aa':
        S = get_aa_scores(adj_train,u2id,v2id)
    if args.model == 'srnmf_cpa':
        S = get_cpa_scores(adj_train,u2id,v2id)
    if args.model == 'srnmf_jc':
        S = get_jc_scores(adj_train,u2id,v2id)
    if args.model == 'srnmf_cn':
        S = get_cn_scores(adj_train,u2id,v2id)
    print('finished computing S')

    X, Y = getXY(S, adj_train, k)
    print('finished computing updating XY')

    B_hat = np.nan_to_num(X@Y)

    test_precision = get_precision(test_edges, test_edges_false, B_hat, adj_orig, sparse_to_tuple(sparse.csr_matrix(train_edges))[0], u2id, v2id)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, B_hat, adj_orig)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap), 
              'test precision=','{:.5f}'.format(test_precision))
    # exit()

    test_roc_list.append(test_roc)
    test_ap_list.append(test_ap)
    test_precision_list.append(test_precision)
    # break

mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))
mean_precision, ste_precision = np.mean(test_precision_list), np.std(test_precision_list)/(args.numexp**(1/2))

print(args.similarity)
print('mean_roc=','{:.5f}'.format(mean_roc),', ste_roc=','{:.5f}'.format(ste_roc))
print('mean_ap=','{:.5f}'.format(mean_ap),', ste_ap=','{:.5f}'.format(ste_ap))
print('mean_precision=','{:.5f}'.format(mean_precision),', ste_precision=','{:.5f}'.format(ste_precision))

roc = '{:.1f}'.format(mean_roc*100.0)+'+'+'{:.2f}'.format(ste_roc*100.0).strip(' ')
ap = '{:.1f}'.format(mean_ap*100.0)+'+'+'{:.2f}'.format(ste_ap*100.0).strip(' ')
prec = '{:.1f}'.format(mean_precision*100.0)+'+'+'{:.2f}'.format(ste_precision*100.0).strip(' ')

print(roc)
print(ap)
print(prec)