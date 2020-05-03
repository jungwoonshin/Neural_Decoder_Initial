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
import args
import model
import pickle

'''
adj, features = load_data_bp_block_adjacency('gpcr')
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all = mask_bipartite_perturbation_test_edges(adj)

with open('u2id.pkl', 'rb') as f:
    u2id = pickle.load(f)
with open('v2id.pkl', 'rb') as f:
    v2id = pickle.load(f)

adj_train_2 = np.load('data/'+str(args.model)+'_feature_matrix.npz')
data = adj_train_2['data']
indices = adj_train_2['indices']
indptr = adj_train_2['indptr']
shape = adj_train_2['shape']
adj_train_2 = sp.csr_matrix((data, indices, indptr), shape=shape)
adj_train_2.tolil().setdiag(np.zeros(adj_train_2.shape[0]))

right_bottom = adj_train_2.toarray()[len(u2id):,len(u2id):]
sum_right_bottom = right_bottom.sum()
average_value_right_bottom = sum_right_bottom / (len(v2id)*len(v2id))

print('sum_right_bottom: ', sum_right_bottom)
print('(len(v2id)*len(v2id)): ', (len(v2id)*len(v2id)))
print('average_value_right_bottom: ', average_value_right_bottom)

print('='*88)

right_bottom_adj = adj.toarray()[len(u2id):,len(u2id):]
sum_right_bottom_adj = right_bottom_adj.sum()
average_value_right_bottom_adj = sum_right_bottom_adj / (len(v2id)*len(v2id))

print('sum_right_bottom_adj: ', sum_right_bottom_adj)
print('(len(v2id)*len(v2id)): ', (len(v2id)*len(v2id)))
print('average_value_right_bottom_adj: ', average_value_right_bottom_adj)

print('='*88)


right_top = adj_train_2.toarray()[:len(u2id),len(u2id):]
sum_right_top = right_top.sum()
average_value_right_top_adj = sum_right_top / (len(v2id)*len(u2id))

print('sum_right_top: ', sum_right_top)
print('(len(v2id)*len(u2id)): ', (len(v2id)*len(u2id)))
print('average_value_right_top_adj: ', average_value_right_top_adj)

print('='*88)

right_top_adj = adj.toarray()[:len(u2id),len(u2id):]
sum_right_top_adj = right_top_adj.sum()
average_value_right_top_adj = sum_right_top_adj / (len(v2id)*len(u2id))

print('sum_right_top_adj: ', sum_right_top_adj)
print('(len(v2id)*len(u2id)): ', (len(v2id)*len(u2id)))
print('average_value_right_top_adj: ', average_value_right_top_adj)


print('='*88)

print('318**2: ', 318**2.0)
whole = len(u2id) * len(u2id) + len(v2id)*len(v2id) + 2*len(v2id)*len(u2id)
print('whole: ' , whole)
'''
# array = np.array([[0,0,0,1,1,0],[0,0,0,0,1,1],[0,0,0,1,1,0],\
# 				 [1,0,1,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0]])


A = np.array([[0,0,1,1,0,1],\
				 [0,0,0,1,1,1],\
				 [1,0,0,0,0,0],\
				 [1,1,0,0,0,0],\
				 [0,1,0,0,0,0],\
				 [1,1,0,0,0,0]])
B = np.array([[.5,.5,1,1,0,1],\
			 [.4,.4,0,1,1,1],\
			 [1,0,0.2,0.2,0.2,0.2],\
			 [1,1,0.3,0.3,0.3,0.3],\
			 [0,1,0.4,0.4,0.4,0.4],\
			 [1,1,0.5,0.5,0.5,0.5]])
A = A.astype(float)
print(A)
# print('AT:',np.transpose(A))
# print(A+np.transpose(A*0.5))
print(B)
# A\ = np.matmul(A,A)
# print('A^2:\n ',A2)
# A = np.matmul(A,A2)
# print('A^3:\n ',A)
# # A = np.matmul(A,A)
# print('A^4:\n', A)
# A = sp.csr_matrix(A)
# print(A.toarray())
# adj_ = A + sp.eye(A.shape[0])
# rowsum = np.array(adj_.sum(1))
# degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
# adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
# print(adj_normalized.toarray())
print(A)
# print(np.matmul(A,np.transpose(A)))
# print(np.matmul(np.transpose(A),A))
print(A @ A)
# print(preprocess_graph_numpy(A))
# A = np.array([[0,0,1,1,0,1],\
# 				 [0,0,0,1,1,1],\
# 				 [0,0,0,0,0,0],\
# 				 [0,0,0,0,0,0],\
# 				 [0,0,0,0,0,0],\
# 				 [0,0,0,0,0,0]])
# print(A+A.T)


# A = A.astype(float)
# B = B.astype(float)
# AB =np.matmul(A,B)
# # print(np.divide(np.matmul(A,B)+np.transpose(np.matmul(A,B)),2))

# # one = np.multiply(np.matmul(A,B), A)
# # two = np.matmul(A,np.multiply(B,A))
# # print(one)
# # print(two)
# # print(AB)
# # print(np.array(preprocess_graph(AB)))
# indexes = np.where(A==0)
# print(indexes[0])
# print(indexes[1])
# indexes = np.where(A[:2,2:]==0)
# # indexes[1] += 2
# print(indexes[0])
# print(np.array(indexes[1])+2)

# np.random.seed(0)
# np.random.shuffle(indexes[0])
# np.random.seed(0)
# np.random.shuffle(indexes[1])
# print(indexes[0])
# print(indexes[1])
# val_index_i = indexes[0][:num_test]
# val_index_j = indexes[1][:num_test]

# test_index_i = indexes[1][num_val:num_test+num_val]
# index_j = indexes[1][num_val:num_test+num_val]