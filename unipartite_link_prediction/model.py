import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation
		self.input_dim = input_dim

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)

		outputs = self.activation(x)

		return outputs

def normal_init(input_dim, output_dim):
	embeddings = nn.Embedding(input_dim, output_dim)
	nn.init.normal_(embeddings.weight, std=0.01)
	return embeddings

def kaiming_init(input_dim, output_dim):
	embeddings = torch.empty(input_dim, output_dim)
	nn.init.kaiming_uniform_(embeddings, a=1, nonlinearity='sigmoid')
	return nn.Parameter(embeddings)

def xavier_init(input_dim, output_dim):
	embeddings = torch.empty(input_dim, output_dim)
	torch.nn.init.xavier_uniform_(embeddings)
	return nn.Parameter(embeddings)

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	np.random.seed(args.weight_seed)
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	args.weight_seed += 1
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self,adj):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X, train_edges, train_false_edges):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class LGAE(nn.Module):
	def __init__(self,adj):
		super(LGAE,self).__init__()
		# self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.layer = GraphConvSparse(args.input_dim, args.hidden1_dim, adj, activation=lambda x:x)

	def encode(self, X):
		z = self.layer(X)
		return z

	def decode(self, z, edges):
		# z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)
		z_elementmult_matrix = torch.zeros(len(edges), 1).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)

			# z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			z_elementmult_matrix[index,:] = (z_first * z_second).sum()

		# z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two
		# z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return torch.sigmoid(z_elementmult_matrix)

	def forward(self, X, train_edges, train_false_edges):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		# edges = np.concatenate((train_edges,train_false_edges),0)
		# A_pred = self.decode(Z, edges)
		return A_pred

# linear 
class NLGF(nn.Module):
	def __init__(self,adj):
		super(NLGF,self).__init__()
		self.layer = Layer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class Layer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(Layer, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.weight_two = glorot_init(output_dim*2, output_dim)
		self.weight_three = glorot_init(output_dim*2, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def encode(self, x):
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)
		return z

	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)
		z_elementmult_matrix = torch.zeros(len(edges), self.output_dim).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)

			# z_first_gmf = z_gmf[i,:].view(1,-1)
			# z_second_gmf = z_gmf[j,:].view(1,-1)

			z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			z_elementmult_matrix[index,:] = z_first * z_second

		z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two
		z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		# z_gmf = self.encode2(x)
		edges = np.concatenate((train_edges,train_false_edges),0)

		result = self.decode(z, edges)
		# z2 = self.decode(z, train_false_edges)
		# result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)

class NGF(nn.Module):
	def __init__(self,adj):
		super(NGF,self).__init__()
		self.layer = GAELayer(args.input_dim, args.hidden1_dim, args.hidden2_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class GAELayer(nn.Module):
	def __init__(self, input_dim, output_dim, output_dim2, adj, **kwargs):
		super(GAELayer, self).__init__(**kwargs)
		self.enc_weight_1 = xavier_init(input_dim, output_dim)
		self.enc_weight_2 = xavier_init(output_dim, output_dim2)
		self.weight_two = xavier_init(output_dim2*2, output_dim2)
		self.weight_three = kaiming_init(output_dim2*2, 1)
		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.output_dim2 = output_dim2

	def encode(self, x):
		x = torch.mm(x,self.enc_weight_1)
		z = torch.mm(self.adj, x)
		z = F.relu(z)

		x = torch.mm(z,self.enc_weight_2)
		z = torch.mm(self.adj, x)
		return z

	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim2*2).to(device)
		z_elementmult_matrix = torch.zeros(len(edges), self.output_dim2).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)

			z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			z_elementmult_matrix[index,:] = z_first * z_second

		z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two
		z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		z1 = self.decode(z, train_edges)
		z2 = self.decode(z, train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)

class NVGF(nn.Module):
	def __init__(self,adj):
		super(NVGF,self).__init__()
		self.layer = VGAELayer(args.input_dim, args.hidden1_dim, args.hidden2_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class VGAELayer(nn.Module):
	def __init__(self, input_dim, output_dim, output_dim2, adj, **kwargs):
		super(VGAELayer, self).__init__(**kwargs)
		self.enc_weight_1 = glorot_init(input_dim, output_dim)
		self.enc_weight_2 = glorot_init(output_dim, output_dim2)
		self.enc_weight_3 = glorot_init(output_dim, output_dim2)
		self.weight_two = glorot_init(output_dim2*2, output_dim2)
		self.weight_three = glorot_init(output_dim2*2, 1)
		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.output_dim2 = output_dim2

	def encode(self, x):
		x = torch.mm(x,self.enc_weight_1)
		z = torch.mm(self.adj, x)
		z = F.relu(z)

		x = torch.mm(z,self.enc_weight_2)
		self.mean = mean = torch.mm(self.adj, x)
		
		x = torch.mm(z,self.enc_weight_3)
		self.logstd = logstd = torch.mm(self.adj, x)
		
		gaussian_noise = torch.randn(x.size(0), args.hidden2_dim).to(device)
		sampled_z = gaussian_noise*torch.exp(logstd) + mean
		return sampled_z


	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim2*2).to(device)
		z_elementmult_matrix = torch.zeros(len(edges), self.output_dim2).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)

			z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			z_elementmult_matrix[index,:] = z_first * z_second

		z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two
		z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		z1 = self.decode(z, train_edges)
		z2 = self.decode(z, train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)

class VGAE2(nn.Module):
	def __init__(self, adj):
		super(VGAE2,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.decoder = Decoder(args.input_dim, args.hidden2_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim).cuda()
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		a_pred = self.decoder(sampled_z, train_edges, train_false_edges)
		return a_pred

	def forward(self, X, train_edges, train_false_edges):
		Z = self.encode(X, train_edges, train_false_edges)
		return Z

class SGAE(nn.Module):
	def __init__(self,adj):
		super(SGAE,self).__init__()
		self.layer = SGAELayer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class SGAELayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(SGAELayer, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.weight_two = glorot_init(int(output_dim), int(output_dim/2))
		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def encode(self, x, activation):
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)
		z = activation(x)
		return z
	def encode2(self, x, activation):
		x = torch.mm(x,self.weight_two)
		z = torch.mm(self.adj, x)
		z = activation(x)
		return z

	def decode(self, z, edges):
		z_elementmult_matrix = torch.zeros(len(edges), 1).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)
			z_elementmult_matrix[index,:] = (z_first * z_second).sum()

		z = z_elementmult_matrix
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x, F.relu)
		z = self.encode2(z, lambda x:x)
		z1 = self.decode(z, train_edges)
		z2 = self.decode(z, train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)


class SLGAE_Concat(nn.Module):
	def __init__(self,adj):
		super(SLGAE_Concat,self).__init__()
		self.layer = SLGAELayer_concat(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class SLGAELayer_concat(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(SLGAELayer_concat, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim) 
		self.weight_two = xavier_init(output_dim*2, 1)
		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def encode(self, x, activation):
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)
		z = activation(x)
		return z

	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)
			z_pair_matrix[index,:] = torch.cat((z_first ,z_second),1)

		z = z_pair_matrix @ self.weight_two
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x, lambda x:x)
		z1 = self.decode(z, train_edges)
		z2 = self.decode(z, train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)


class SLGAE_Innerproduct(nn.Module):
	def __init__(self,adj):
		super(SLGAE_Innerproduct,self).__init__()
		self.layer = SLGAELayer_Innerproduct(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class SLGAELayer_Innerproduct(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(SLGAELayer_Innerproduct, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim) 
		self.weight_two = xavier_init(output_dim, 1)
		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def encode(self, x, activation):
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)
		z = activation(x)
		return z

	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)
			z_pair_matrix[index,:] =z_first *z_second

		z = z_pair_matrix @ self.weight_two
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x, lambda x:x)
		z1 = self.decode(z, train_edges)
		z2 = self.decode(z, train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)
