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
from networkx.algorithms import bipartite
import pickle 
import collections
import matplotlib.pyplot as plt
from collections import OrderedDict
from random import shuffle
import random

import args

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset, change_seed=False):
    
    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    u_list = []
    v_list = []
    for line in f:
         # malaria case
        if dataset == 'malaria':
            edge = line.strip('\n').split('\t')
            edge1 = edge[1]
            edge2 = edge[2]
            u_list.append(edge1)
            v_list.append(edge2)
            continue 
        elif dataset == 'movie100k':
            edge = line.strip('\n').split('\t')
            edge1 = edge[0]
            edge2 = edge[1]
            u_list.append(edge1)
            v_list.append(edge2)
            continue
        elif len(line.strip('\n').split('\t')) > 1:
            edge = line.strip('\n').split('\t')
        elif len(line.strip('\n').split(' ')) > 1:
            edge = line.strip('\n').split(' ')
        elif len(line.strip('\n').split('::')) > 1:
            edge = line.strip('\n').split('::')
        else:
            print('error while preprocessing data. exiting...')
            exit()
        edge1 = edge[0]
        edge2 = edge[1]
        u_list.append(edge1)
        v_list.append(edge2)

    u_list= list( OrderedDict.fromkeys(u_list) )
    v_list = list( OrderedDict.fromkeys(v_list) )
    
    random.seed(0)
    shuffle(u_list)
    random.seed(0)
    shuffle(v_list)

    id2u = OrderedDict()
    u2id = OrderedDict()
    for index, val in enumerate(u_list):
        id2u[index] = val
        u2id[val] = index

    id2v = {}
    v2id = {}
    for index, val in enumerate(v_list):
        index = index+len(u_list)
        id2v[index] = val
        v2id[val] = index

    with open('data/bipartite/id2name/'+ str(args.dataset) +'id2v.pkl', 'wb') as f:
        pickle.dump(id2v, f)
    with open('data/bipartite/id2name/'+ str(args.dataset) +'id2u.pkl', 'wb') as f:
        pickle.dump(id2u, f)
    with open('data/bipartite/id2name/'+ str(args.dataset) +'u2id.pkl', 'wb') as f:
        pickle.dump(u2id, f)
    with open('data/bipartite/id2name/'+ str(args.dataset) +'v2id.pkl', 'wb') as f:
        pickle.dump(v2id, f)

    f = open("data/bipartite/edgelist_"+dataset+"_bp", 'r')
    graph = defaultdict(list)

    edge_list = []
    for line in f:
        # malaria case
        if dataset == 'malaria':
            edge = line.strip('\n').split('\t')
            edge1 = u2id[edge[1]]
            edge2 = v2id[edge[2]]
            edge_list.append((edge1,edge2))
            graph[edge1].append(edge2)
            continue 
        elif dataset == 'movie100k':
            edge = line.strip('\n').split('\t')
            if int(edge[2]) < 3.0:
                graph[u2id[edge[0]]]
                graph[v2id[edge[1]]]
                continue
        elif len(line.strip('\n').split('\t')) > 1:
            edge = line.strip('\n').split('\t')
        elif len(line.strip('\n').split(' '))> 1:
            edge = line.strip('\n').split(' ')
        else:
            edge = line.strip('\n').split('::')
        edge1 = u2id[edge[0]]
        edge2 = v2id[edge[1]]
        edge_list.append((edge1,edge2))
        graph[edge1].append(edge2)

    graph_dict_if_lists = graph
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_if_lists))

    # check if 0 is value is set on top-right and bottom-left side of adjacency matrix.
    warning=0
    warning += adj[:len(u2id),:len(u2id)].sum()
    warning += adj[len(u2id):,len(u2id):].sum()
    print('number of elements that are not 0: ', warning)
    
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

def load_data_drug(dataset, change_seed=False):
    with open('data/drug_graph.pickle', 'rb') as f:
        drug_graph = pickle.load(f)
    with open('data/drug_adjacency.pickle', 'rb') as f:
        adj = pickle.load(f)

    file = open('data/drug_adjacency matrix.txt')
    next(file)
    u_name_list = []
    for line in file:
        u_name = line.strip('\n').split('\t')[0]
        u_name_list.append(u_name)

    with open('data/bipartite/id2name/'+ str(args.dataset) +'u2id.pkl', 'wb') as f:
        pickle.dump(u_name_list, f)

    v_name_list = range(adj.shape[0]-len(u_name_list))
    with open('data/bipartite/id2name/'+ str(args.dataset)  +'v2id.pkl', 'wb') as f:
        pickle.dump(v_name_list, f)

    total_number_nodes = adj.shape[0]
    print(total_number_nodes)
    row = np.array([x for x in range(total_number_nodes)])
    col = np.array([x for x in range(total_number_nodes)])
    data = np.array([1 for x in range(total_number_nodes)])
    features = csr_matrix((data, (row, col)), shape=(total_number_nodes, total_number_nodes)).tolil()
    
    print(adj.shape)
    print(features.shape)
    return adj, features

def load_data_toy(dataset, change_seed=False):

    adj = np.array([[0,0,0,1,1,1,1,0,1],\
                    [0,0,0,1,1,0,1,1,1],\
                    [0,0,0,1,1,0,0,1,0],\
                    [0,0,0,1,1,0,1,0,1],\
                    [0,0,0,0,0,0,0,0,0],\
                    [0,0,0,0,0,0,0,0,0],\
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0]])
    adj = np.random.randint(2, size=(15,15))
    adj[:5,:5] = np.zeros((5,5))
    adj[5:,5:] = np.zeros((10,10))
    adj[5:,:5] = np.zeros((10,5))
    adj += adj.T
    
    adj = sp.csr_matrix(adj)

    u_name_list = [1,2]
    v_name_list = [3,4,5,6]

    with open('data/bipartite/id2name/'+ str(args.dataset) +'u2id.pkl', 'wb') as f:
        pickle.dump(u_name_list, f)

    v_name_list = range(adj.shape[0]-len(u_name_list))
    with open('data/bipartite/id2name/'+ str(args.dataset)  +'v2id.pkl', 'wb') as f:
        pickle.dump(v_name_list, f)

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
