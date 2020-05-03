'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''

import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from collections import defaultdict
from statistics import mean

from postprocessing import *
from preprocessing import *
from input_data import *

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_fd_neg(dataset):
    with open("data/fd_neg/id2disease.pickle", 'rb') as f:
        id2disease = pkl.load(f)
    with open("data/fd_neg/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pkl.load(f)

    total_number_nodes = len(id2disease) + len(id2ingredient)
    print('total_number_nodes:',total_number_nodes)

    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    whole_x = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))
    # print('x:',x)
    idx = int(whole_x.shape[0]*0.9)

    x = whole_x[:idx,:]
    tx = whole_x[idx:,:]
    allx = x

    f = open("data/fd_neg/edgelist_neg", 'r')
    graph = defaultdict(list)
    number_of_edges = 0
    for line in f:
        number_of_edges += 1
        edge = line.strip('\n').split(' ')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)

    features = sp.vstack((allx, tx)).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    G=nx.from_dict_of_lists(graph)
    degree_of_node = list(G.degree(G.nodes()))
    degree_of_node = [b for (a,b) in degree_of_node]
    degree_of_node = mean(degree_of_node)
    print('degree_of_node:',degree_of_node)
    print('number of edges: ', len(G.edges()))
    print('number of nodes: ', len(G.nodes()))

    # row = np.array([x for x in range(total_number_nodes)])
    # col = np.array([x for x in range(total_number_nodes)])
    # data = np.random.uniform(low=-1, high=1, size=(total_number_nodes,))
    # features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))

    print(adj.shape)
    print(features.shape)

    return adj, features
def load_data_fd(dataset):
    with open("data/fd/id2disease.pickle", 'rb') as f:
        id2disease = pkl.load(f)
    with open("data/fd/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pkl.load(f)

    total_number_nodes = len(id2disease) + len(id2ingredient)
    print('total_number_nodes:',total_number_nodes)

    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    whole_x = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))
    # print('x:',x)
    idx = int(whole_x.shape[0]*0.9)

    x = whole_x[:idx,:]
    tx = whole_x[idx:,:]
    allx = x

    f = open("data/fd/edgelist", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split(' ')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)

    features = sp.vstack((allx, tx)).tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


    G=nx.from_dict_of_lists(graph)
    degree_of_node = list(G.degree(G.nodes()))
    degree_of_node = [b for (a,b) in degree_of_node]
    degree_of_node = mean(degree_of_node)
    print('degree_of_node:',degree_of_node)
    print('number of edges: ', len(G.edges()))
    print('number of nodes: ', len(G.nodes()))
    # print(type(adj))
    # print((adj!=adj.T).nnz==0)

    # row = np.array([x for x in range(total_number_nodes)])
    # col = np.array([x for x in range(total_number_nodes)])
    # data = np.random.uniform(low=-1, high=1, size=(total_number_nodes,))
    # features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes))

    print(adj.shape)
    print(features.shape)

    return adj, features

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    # print('x:',x.shape)
    # print('tx:',tx.shape)
    # print('allx:',allx)
    # print('graph:',graph)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # total_number_nodes = len(graph.keys())
    # print(total_number_nodes)
    # row = np.array([x for x in range(total_number_nodes)])
    # col = np.array([x for x in range(total_number_nodes)])
    # data = np.array([1 for x in range(total_number_nodes)])
    # features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features

def load_data_featureless(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    # print('x:',x.shape)
    # print('tx:',tx.shape)
    # print('allx:',allx)
    # print('graph:',graph)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    # features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features



def load_data_cora_bp(dataset):

    f = open("data/bipartite/edgelist_cora_bp", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split('\t')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # features = sp.vstack((allx, tx)).tolil()

    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    
    print(adj.shape)
    print(features.shape)


    G=nx.from_dict_of_lists(graph)
    degree_of_node = list(G.degree(G.nodes()))
    degree_of_node = [b for (a,b) in degree_of_node]
    degree_of_node = mean(degree_of_node)
    print('degree_of_node:',degree_of_node)
    print('number of edges: ', len(G.edges()))
    print('number of nodes: ', len(G.nodes()))
    # exit()
    return adj, features


def load_data_pubmed_bp(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    f = open("data/bipartite/edgelist_pubmed_bp", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split('\t')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # features = sp.vstack((allx, tx)).tolil()

    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    
    print(adj.shape)
    print(features.shape)
    # exit()
    return adj, features


def load_data_citeseer_bp(dataset):

    f = open("data/bipartite/edgelist_citeseer_bp", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split('\t')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # features = sp.vstack((allx, tx)).tolil()
    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]

    print(adj.shape)
    print(features.shape)


    G=nx.from_dict_of_lists(graph)
    degree_of_node = list(G.degree(G.nodes()))
    degree_of_node = [b for (a,b) in degree_of_node]
    degree_of_node = mean(degree_of_node)
    print('degree_of_node:',degree_of_node)
    print('number of edges: ', len(G.edges()))
    print('number of nodes: ', len(G.nodes()))
    # exit()
    return adj, features


def load_data_pubmed_bp(dataset):

    f = open("data/bipartite/edgelist_pubmed_bp", 'r')
    graph = defaultdict(list)
    for line in f:
        edge = line.strip('\n').split('\t')
        graph[int(edge[0])].append(int(edge[1]))
    # print(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # features = sp.vstack((allx, tx)).tolil()
    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]

    print(adj.shape)
    print(features.shape)


    G=nx.from_dict_of_lists(graph)
    degree_of_node = list(G.degree(G.nodes()))
    degree_of_node = [b for (a,b) in degree_of_node]
    degree_of_node = mean(degree_of_node)
    print('degree_of_node:',degree_of_node)
    print('number of edges: ', len(G.edges()))
    print('number of nodes: ', len(G.nodes()))
    print('number of edges percentage: ', len(G.edges())/(len(G.nodes())*len(G.nodes())))

    # exit()
    return adj, features

def check_implicit_connection():
    
    adj, features,\
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = get_data(args.dataset)

    with open('u2id.pkl', 'rb') as f:
        u2id = pickle.load(f)
    with open('v2id.pkl', 'rb') as f:
        v2id = pickle.load(f)

    learned_feature = np.load('data/'+str(args.model1) +'_' +str(args.dataset)+'_reconstructed_matrix.npz')
    data = learned_feature['data']
    indices = learned_feature['indices']
    indptr = learned_feature['indptr']
    shape = learned_feature['shape']
    reconstructed_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
    reconstructed_matrix.tolil().setdiag(np.zeros(reconstructed_matrix.shape[0]))
    reconstructed_matrix = preprocess_graph(reconstructed_matrix)

    reconstructed_matrix = torch.sparse.FloatTensor(torch.LongTensor(reconstructed_matrix[0].T), 
                                torch.FloatTensor(reconstructed_matrix[1]), 
                                torch.Size(reconstructed_matrix[2]))
    reconstructed_matrix = reconstructed_matrix.to_dense().numpy()


    adj_orig = adj  
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))
    
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_sparse_norm = adj_norm
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))

    adj_unnormalized = sp.coo_matrix(adj)
    adj_unnormalized = sparse_to_tuple(adj_unnormalized)
    adj_unnormalized = torch.sparse.FloatTensor(torch.LongTensor(adj_unnormalized[0].T), 
                                torch.FloatTensor(adj_unnormalized[1]), 
                                torch.Size(adj_unnormalized[2]))
    print(type(np.array(adj_sparse_norm)))
    print(type(reconstructed_matrix))

    # adj_sparse_norm = adj_norm.to_dense().numpy()
    # reconstructed_matrix = reconstructed_matrix.astype(float)
    # AX = np.matmul(adj_sparse_norm, reconstructed_matrix)
    # print(AX)
    # pos_val=0.0
    # neg_val=0.0

    # pos_val_2 = 0.0
    # neg_val_2 = 0.0

    # pos_val_3 = 0.0
    # neg_val_3 = 0.0
    # for (i,j) in test_edges:
    #     pos_val+=AX[i,j]
    #     # pos_val_2 +=reconstructed_matrix[i,j]
    # for (i,j) in test_edges_false:
    #     neg_val+=AX[i,j]

    # for (i,j) in val_edges:
    #     pos_val_2+=AX[i,j]
    #     # pos_val_2 +=reconstructed_matrix[i,j]
    # for (i,j) in val_edges_false:
    #     neg_val_2+=AX[i,j]
    #     # neg_val_2 += reconstructed_matrix[i,j]

    # for (i,j) in edges_all:
    #     pos_val_3+=AX[i,j]
    #     # pos_val_2 +=reconstructed_matrix[i,j]
    # for (i,j) in edges_false_all:
    #     neg_val_3+=AX[i,j]

    # print(pos_val/len(test_edges))
    # print(neg_val/len(test_edges_false))
    # print(pos_val_2/len(val_edges))
    # print(neg_val_2/len(val_edges_false))
    # print(pos_val_3/len(edges_all))
    # print(neg_val_3/len(edges_false_all))
    # print(AX.sum()/(AX.shape[0]*AX.shape[1]))

    adj_sparse_norm = adj_norm.to_dense().numpy()
    # reconstructed_matrix = reconstructed_matrix.astype(float)

    reconstructed_matrix_first = reconstructed_matrix.copy()
    print(len(u2id))
    print(len(v2id))
    reconstructed_matrix_first[:len(u2id),:len(u2id):] = np.zeros((len(u2id),len(u2id)))
    reconstructed_matrix_first[len(u2id):,len(u2id):] = np.zeros((len(v2id),len(v2id)))

    AX = np.matmul(reconstructed_matrix_first, reconstructed_matrix)
    AX = (AX+ np.transpose(AX))/2.0
    # print(reconstructed_matrix.shape)
    # print(reconstructed_matrix[:len(u2id),len(u2id):].sum())
    # print(reconstructed_matrix[len(u2id):,:len(u2id)].sum())
    # exit()

    pos_val=0.0
    neg_val=0.0

    pos_val_2 = 0.0
    neg_val_2 = 0.0

    pos_val_3 = 0.0
    neg_val_3 = 0.0
    for (i,j) in test_edges:
        pos_val+=AX[i,j]
        # pos_val_2 +=reconstructed_matrix[i,j]
    for (i,j) in test_edges_false:
        neg_val+=AX[i,j]

    for (i,j) in val_edges:
        pos_val_2+=AX[i,j]
        # pos_val_2 +=reconstructed_matrix[i,j]
    for (i,j) in val_edges_false:
        neg_val_2+=AX[i,j]
        # neg_val_2 += reconstructed_matrix[i,j]

    for (i,j) in edges_all:
        pos_val_3+=AX[i,j]
        # pos_val_2 +=reconstructed_matrix[i,j]
    for (i,j) in edges_false_all:
        neg_val_3+=AX[i,j]

    print(pos_val/len(test_edges))
    print(neg_val/len(test_edges_false))
    print(pos_val_2/len(val_edges))
    print(neg_val_2/len(val_edges_false))
    print(pos_val_3/len(edges_all))
    print(neg_val_3/len(edges_false_all))

        
def merge(dict1, dict2): 
    return {**dict1, **dict2}

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol) 
def check_2hop_connection():
    # args.dataset = 'fdpos'

    learned_feature = np.load('data/'+str(args.dataset)+'_reconstructed_matrix.npz')
    data = learned_feature['data']
    indices = learned_feature['indices']
    indptr = learned_feature['indptr']
    shape = learned_feature['shape']
    reconstructed_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
    reconstructed_matrix = preprocess_graph(reconstructed_matrix)

    reconstructed_matrix = torch.sparse.FloatTensor(torch.LongTensor(reconstructed_matrix[0].T), 
                                torch.FloatTensor(reconstructed_matrix[1]), 
                                torch.Size(reconstructed_matrix[2]))
    reconstructed_matrix = reconstructed_matrix.to_dense().numpy()

    adj, features,\
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = get_data(args.dataset)

    adj_orig = adj  
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))
    
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_sparse_norm = adj_norm
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))

    adj_unnormalized = sp.coo_matrix(adj)
    adj_unnormalized = sparse_to_tuple(adj_unnormalized)
    adj_unnormalized = torch.sparse.FloatTensor(torch.LongTensor(adj_unnormalized[0].T), 
                                torch.FloatTensor(adj_unnormalized[1]), 
                                torch.Size(adj_unnormalized[2]))
    print(type(np.array(adj_sparse_norm)))
    print(type(reconstructed_matrix))
    adj_sparse_norm = adj_norm.to_dense().numpy()
    reconstructed_matrix = reconstructed_matrix.astype(float)

    # print(check_symmetric(reconstructed_matrix))
    # exit()
    # adj_sparse_norm = adj_sparse_norm.astype(float)
    AX = np.matmul(adj_sparse_norm, reconstructed_matrix)
    AX = (AX + np.transpose(AX))/2.0
    print(check_symmetric(AX))


    test_roc, test_ap = get_scores(test_edges, test_edges_false, AX, adj_orig)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
              "test_ap=", "{:.5f}".format(test_ap))


    exit()
    with open("data/fd/id2disease.pickle", 'rb') as f:
        id2disease = pkl.load(f)
    with open("data/fd/id2ingredient.pickle", 'rb') as f:
        id2ingredient = pkl.load(f)
    with open("data/fd/disease2id.pickle", 'rb') as f:
        disease2id = pkl.load(f)
    with open("data/fd/ingredient2id.pickle", 'rb') as f:
        ingredient2id = pkl.load(f)

    entity2id = merge(disease2id, ingredient2id)
    id2entity = merge(id2disease, id2ingredient)

    name1 = 'obesity'
    name2 = 'milk'

    # from_node_id = cancer_id = disease2id['colon cancer']
    # to_node_id = ingredient_id = ingredient2id['meat']
    from_node_id = cancer_id = disease2id[name1]
    to_node_id = ingredient_id = ingredient2id[name2]
    # print(adj_sparse_norm[cancer_id,:])
    # print(reconstructed_matrix[:, ingredient_id])


    for index in range(adj_sparse_norm.shape[0]):
        connectivity1 = adj_sparse_norm[cancer_id,:][index]
        connectivity2 = reconstructed_matrix[:, ingredient_id][index] 

        if connectivity1 != 0 and connectivity2 != 0 :
            middle_node_id = index
            # print(id2entity)
            # print(middle_node_id)
            middle_entity_name = id2entity[middle_node_id]
            connectivity1 *= reconstructed_matrix[:, ingredient_id][index]
            print(name1 + ' => ',middle_entity_name, ' => '+name2+': ', connectivity1)

    print()
    adj_sparse_norm_copy = adj_sparse_norm
    adj_sparse_norm = reconstructed_matrix
    reconstructed_matrix = adj_sparse_norm_copy

    for index in range(adj_sparse_norm.shape[0]):
        connectivity1 = adj_sparse_norm[cancer_id,:][index]
        connectivity2 = reconstructed_matrix[:, ingredient_id][index] 

        if connectivity1 != 0 and connectivity2 != 0 :
            middle_node_id = index
            # print(id2entity)
            # print(middle_node_id)
            middle_entity_name = id2entity[middle_node_id]
            connectivity1 *= reconstructed_matrix[:, ingredient_id][index]
            print(name1 + ' => ',middle_entity_name, ' => '+name2+': ', connectivity1)






def main():
    # adj, features = load_data_fd_neg('1')
    # adj, features = load_data_fd('cora')
    # adj, features = load_data('cora')
    # adj, features = load_data_fd('1')
    # adj, features = load_data_cora_bp('1')
    # adj, features = load_data_citeseer_bp('1')
    # load_data_pubmed_bp('1')

    check_implicit_connection()
    # check_2hop_connection()

    
main()