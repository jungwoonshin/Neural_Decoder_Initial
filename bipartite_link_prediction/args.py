### CONFIGS ###
model = 'VGAE'

#####################################
# 		 select one dataset 		#
#####################################
# dataset = 'gpcr'
# dataset='enzyme'
# dataset = 'ionchannel'
# dataset= 'malaria'
# dataset = 'drug'
# dataset = 'sw'
# dataset = 'nanet'
dataset = 'movie100k'

device = '1'

#model related parameter
hidden1_dim = 32
hidden2_dim = 16
subsample_number = 5
learning_rate = 0.01
num_epoch = 200

numexp = 10
num_test = 10./1. # 10/1 means 10% means 10% is used as test sets., 10/2 means 20% is used as test sets. 
print_val = True
weight_seed = 100
edge_idx_seed = 100
edge_idx_seed_2 = 200
edge_idx_seed_3 = 300

# input_dim1 = 318 # gpcr bipartite
# input_dim1 = 1109 # enzyme bipartite
# input_dim1 = 414 # ion channel bip
# input_dim1 = 1103 # malaria
# input_dim1 = 350 # drug
# input_dim1 = 32 # southernwomen
# input_dim1 = 1880 # nanet
# input_dim1 = 2625 # movie100k