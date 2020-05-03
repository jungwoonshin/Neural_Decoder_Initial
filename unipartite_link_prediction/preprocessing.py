'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
import args

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_graph_neg_one(adj):
    adj_ = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0], adj.shape[1])
    rowsum = np.array(adj_.sum(1))
    np.seterr(divide='ignore')
    rowsum_ = np.power(rowsum, -1).flatten()
    rowsum_[rowsum_ == np.inf] = 0.0
    degree_mat_inv_sqrt = sp.diags(rowsum_)
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
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
    adj_train = adj_train + adj_train.T
    
    # adj_train_copy = adj_train.copy().toarray()
    # iu1 = np.tril_indices(adj_train_copy.shape[0]) 
    # adj_train_copy[iu1] = -1

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

    indexes_train = np.where(adj_train.toarray()==0)

    np.random.seed(args.edge_idx_seed_2)
    np.random.shuffle(indexes_train[0])
    np.random.seed(args.edge_idx_seed_2)
    np.random.shuffle(indexes_train[1])
    args.edge_idx_seed_2 += 1

    # train_index_i = indexes_train[0]
    # train_index_j = np.array(indexes_train[1])
    train_index_i = indexes_train[0][:len(train_edges)+adj_train.shape[0]] # sample more than train edges + diagonal
    train_index_j = indexes_train[1][:len(train_edges)+adj_train.shape[0]]

    whole_adj = adj.toarray()
    diagonal = np.where(np.eye(whole_adj.shape[0],dtype=bool))
    diagonal = set([(x,y) for x,y in zip(diagonal[0],diagonal[1])])

    indexes = np.where(whole_adj==0.0)

    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(indexes[0])
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(indexes[1])

    indexes_0 = indexes[0][:num_test+num_val+len(diagonal)]
    indexes_1 = indexes[1][:num_test+num_val+len(diagonal)]
    indexes = set([(x,y) for x,y in zip(indexes_0,indexes_1)])
    
    indexes = list(indexes - diagonal)
    val_index = indexes[:num_val]
    test_index = indexes[num_val:num_test+num_val]
    
    train_indexes = set([(x,y) for x,y in zip(train_index_i,train_index_j)])
    train_index = list(train_indexes - diagonal)
    train_index = train_index[:len(train_edges)]

    np.random.seed(args.edge_idx_seed_2)
    np.random.shuffle(train_index)
    
    train_false_edges = []
    for x,y in train_index:
        train_false_edges.append([x,y])
    
    val_edges_false = []
    for x,y in val_index:
        val_edges_false.append([x,y])
    
    test_edges_false = []
    for x,y in test_index:
        test_edges_false.append([x,y])

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
    print('len(val_edges_false):', len(val_edges_false))
    print('len(test_edges_false):', len(test_edges_false))
    print('len(edges_all):', len(edges_all))
    print('len(train_false_edges):', len(train_false_edges))

    return adj_train, train_edges, train_false_edges, val_edges, val_edges_false, test_edges, test_edges_false