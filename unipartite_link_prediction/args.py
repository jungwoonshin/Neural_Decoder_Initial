### CONFIGS ###
dataset = 'cora'
# dataset = 'citeseer'
# dataset = 'pubmed'

model = 'NLGF' # NLGF, NGF, NVGF
# model = 'WholeNLGF' # 'WholeNLGF,SLGAE
num_epoch = 200

if dataset == 'cora':
	node_size = 2708
	input_dim = 1433
	subsample_number = 100
	learning_rate = 0.01

if dataset == 'citeseer':
	node_size = 3327
	input_dim = 3703
	subsample_number = 100
	learning_rate = 0.01

if dataset == 'pubmed':
	node_size = 19717
	input_dim = 500
	subsample_number = 500
	learning_rate = 0.05

num_test = 10./1

hidden1_dim = 64
hidden2_dim = 16

weight_seed = 100
edge_idx_seed = 100
edge_idx_seed_2 = 200
edge_idx_seed_3 = 300