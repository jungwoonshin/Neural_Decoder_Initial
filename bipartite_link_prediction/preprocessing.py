'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
import pickle
import torch
import args
import time

from input_data import *
from preprocessing import *

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def preprocess_graph_neg_one(adj):
    adj_ = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    np.seterr(divide='ignore')
    rowsum_ = np.power(rowsum, -1).flatten()
    rowsum_[rowsum_ == np.inf] = 0.0
    degree_mat_inv_sqrt = sp.diags(rowsum_)    
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()

    return sparse_to_tuple(adj_normalized)

def preprocess_graph_numpy(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.toarray()
    
def getMaskForBiAdjanceyMatrix():
    top_left = np.zeros((len(u2id),len(u2id)))
    bottom_left = np.ones((len(v2id),len(u2id)))
    left = np.concatenate((top_left, bottom_left))

    top_right = np.ones((len(u2id),len(v2id)))
    bottom_right = np.zeros((len(v2id),len(v2id)))
    right = np.concatenate((top_right, bottom_right))

    adj_unnormalized = np.concatenate((left, right),axis=1)
    adj_unnormalized = sp.csr_matrix(adj_unnormalized)
    adj_unnormalized = sparse_to_tuple(adj_unnormalized)
    adj_unnormalized = torch.sparse.FloatTensor(torch.LongTensor(adj_unnormalized[0].T), 
                                torch.FloatTensor(adj_unnormalized[1]), 
                                torch.Size(adj_unnormalized[2]))
    return adj_unnormalized


def mask_bipartite_perturbation_test_edges(adj):
    print('args.dataset: ', args.dataset)
    with open('data/bipartite/id2name/'+ str(args.dataset) +'u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open('data/bipartite/id2name/'+ str(args.dataset) +'v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    ''' original training/test'''
    num_test = int(np.floor(edges.shape[0] / args.num_test))
    num_val = int(np.floor(edges.shape[0] / 20.))
    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(all_edge_idx)
    args.edge_idx_seed += 1
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    def isSetValidMember(a,b):
        setA = set()
        setB = set()

        for (x,y) in a:
            setA.add((x,y))
        for (x,y) in b:
            setA.add((x,y))
        return len(setA.intersection(setB)) > 0

    def isSetMember(a,b):
        setA = set()
        setB = set()

        for (x,y) in a:
            setA.add((x,y))
        for index in range(b.shape[0]):
            setB.add((b[index,0],b[index,1]))
        return len(setA.intersection(setB)) > 0

    test_edges_false = []
    val_edges_false = []

    ''' only for large datasets '''
    # if args.dataset == 'movie1m' or args.dataset == 'movie100k' or args.dataset == 'pubmed' or args.dataset == 'nanet':

    top_right_adj = adj[:len(u2id),len(u2id):].toarray()
    indexes = np.where(top_right_adj==0.0)
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(indexes[0])
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(indexes[1])
    
    top_right_train_adj = adj_train[:len(u2id),len(u2id):].toarray()
    # top_right_train_adj = adj_train.toarray()

    indexes_train = np.where(top_right_train_adj==0.0)
    np.random.seed(args.edge_idx_seed_2)
    np.random.shuffle(indexes_train[0])
    np.random.seed(args.edge_idx_seed_2)
    np.random.shuffle(indexes_train[1])
    args.edge_idx_seed_2 += 1

    train_index_i = indexes_train[0][:adj_train.shape[0]+len(train_edges)]
    train_index_j = np.array(indexes_train[1][:adj_train.shape[0]+len(train_edges)]) + len(u2id)
    # train_index_j = np.array(indexes_train[1][:adj_train.shape[0]+len(train_edges)])

    # train_index_i = indexes_train[0]
    # train_index_j = np.array(indexes_train[1]) + len(u2id)

    val_index_i = indexes[0][:num_val]
    val_index_j = np.array(indexes[1][:num_val]) + len(u2id)

    test_index_i = indexes[0][num_val:num_test+num_val]
    test_index_j =  np.array(indexes[1][num_val:num_test+num_val]) + len(u2id)

    train_indexes = set([(x,y) for x,y in zip(train_index_i,train_index_j)])
    val_indexes =  set([(x,y) for x,y in zip(val_index_i,val_index_j)])
    test_indexes =  set([(x,y) for x,y in zip(test_index_i,test_index_j)])
    
    train_index = list(train_indexes)

    np.random.seed(args.edge_idx_seed_2)
    np.random.shuffle(train_index)

    train_false_edges = []
    # for x,y in train_index:
    for x,y in train_index[:len(train_edges)]:
        train_false_edges.append([x,y])

    for i in range(len(val_edges)):
        idx_i = val_index_i[i]
        idx_j = val_index_j[i]
        val_edges_false.append([idx_i, idx_j])

    for i in range(len(test_edges)):
        idx_i = test_index_i[i]
        idx_j = test_index_j[i]
        test_edges_false.append([idx_i, idx_j])

    assert ~isSetMember(test_edges_false, edges)
    print('~isSetMember(test_edges_false, edges) is True')
    assert ~isSetMember(val_edges_false, edges)
    print('~isSetMember(val_edges_false, edges) is True')
    assert ~isSetMember(val_edges, train_edges)
    print('~isSetMember(val_edges, train_edges) is True')
    assert ~isSetMember(test_edges, train_edges)
    print('~isSetMember(test_edges, train_edges) is True')
    assert ~isSetMember(val_edges, test_edges)
    print('~isSetMember(val_edges, test_edges) is True')
    assert ~isSetValidMember(val_edges_false, test_edges_false)
    print('~isSetMember(val_edges_false, test_edges_false) is True')
    
    print('len(train_edges): ',len(train_edges))
    print('len(val_edges): ',len(val_edges))
    print('len(test_edges): ',len(test_edges))
    print('len(edges): ', len(edges))
    print('len(train_false_edges): ',len(train_false_edges))
    print('len(val_edges_false):', len(val_edges_false))
    print('len(test_edges_false):', len(test_edges_false))
    print('len(edges_all):', len(edges_all))
    # print('train false edges!')
    return adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all 


def get_data(dataset):
    if dataset == 'ionchannel' or args.dataset == 'enzyme' or args.dataset == 'gpcr' or \
        args.dataset == 'movie100k' or args.dataset == 'sw' or args.dataset == 'movie1m' or \
        args.dataset =='malaria' or args.dataset=='nanet' or args.dataset == 'c2o':
        adj, features = load_data(args.dataset)
        adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all  = mask_bipartite_perturbation_test_edges(adj)
    
    if dataset == 'drug':
        adj, features = load_data_drug(args.dataset)
        adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all  = mask_bipartite_perturbation_test_edges(adj)

    return adj, features,\
            adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all 