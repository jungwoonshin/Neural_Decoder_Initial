import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np

import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred


def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	torch.manual_seed(args.weight_seed)
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	args.weight_seed += 1

	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self, adj, unnormalized_adj):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(adj.shape[0], args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

def xavier_init(input_dim, output_dim):
	embeddings = torch.empty(input_dim, output_dim)
	torch.nn.init.xavier_uniform_(embeddings)
	return nn.Parameter(embeddings)

class LGAE(nn.Module):
	def __init__(self, adj, unnormalized_adj):
		super(LGAE,self).__init__()
		self.gcn_mean = GraphConvSparse(adj.shape[0], args.hidden1_dim, adj, activation=lambda x:x)
	
	def encode(self, X):
		z = self.gcn_mean(X)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

def swish(x):
    return x * torch.sigmoid(1.4*x)

class NeuralEncoderDecoder(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(NeuralEncoderDecoder, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim)
		self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three =nn.Parameter(torch.empty(output_dim*2,1))
		nn.init.kaiming_uniform_(self.weight_three, a=1, nonlinearity='sigmoid')
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

			z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			z_elementmult_matrix[index,:] = z_first * z_second

		z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two

		z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return z

	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		edges = np.concatenate((train_edges, train_false_edges), 0)
		result = self.decode(z, edges)
		return torch.sigmoid(result)

class ND_LGAE(nn.Module):
	def __init__(self,adj):
		super(ND_LGAE,self).__init__()
		self.layer = NeuralEncoderDecoder(adj.shape[0], args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(adj.shape[0], args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim).to(device)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

class WholeNeuralEncoderDecoder(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(WholeNeuralEncoderDecoder, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.weight_two = glorot_init(output_dim*2, output_dim)
		self.weight_three = glorot_init(output_dim*2, 1)
		self.adj = adj
		self.activation = activation
		self.output_dim = output_dim

	def forward(self, inputs, train_edges, train_false_edges):
		x = inputs
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)

		z1 = z.repeat(1,z.shape[0]).reshape(-1,z.shape[1])
		z2 = z.repeat(z.shape[0],1).reshape(-1,z.shape[1])
		z_pair = F.relu(torch.cat((z1,z2),1))
		z_elementmult = z1 * z2

		z_pair = z_pair @ self.weight_two
		z = torch.cat((z_pair, z_elementmult),1) @ self.weight_three
		z = z.reshape(self.adj.shape[0], self.adj.shape[0])
		outputs = torch.sigmoid(z)
		return outputs


class WHOLE_ND_LGAE(nn.Module):
	def __init__(self,adj):
		super(WHOLE_ND_LGAE,self).__init__()
		self.base_gcn = WholeNeuralEncoderDecoder(adj.shape[0], args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		hidden = self.base_gcn(X, train_edges, train_false_edges)
		return hidden

	def forward(self, X, train_edges, train_false_edges):
		Z = self.encode(X, train_edges, train_false_edges)
		return Z