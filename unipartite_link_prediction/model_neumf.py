import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

class NEUMF_Sample(nn.Module):
	def __init__(self,adj):
		super(NEUMF_Sample,self).__init__()
		self.layer = NEUMF_SampleLayer(args.node_size, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class NEUMF_SampleLayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(NEUMF_SampleLayer, self).__init__(**kwargs)
		self.z = normal_init(input_dim, output_dim)
		self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim*2, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def decode(self, edges):
		z_elementmult_matrix = torch.zeros(len(edges), self.output_dim).to(device)
		z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)

		for index, (i,j) in enumerate(edges):
			print(i,j)
			index_i = torch.LongTensor([i]).to(device)
			index_j = torch.LongTensor([j]).to(device)

			z_first = self.z(index_i).view(1,-1)
			z_second = self.z(index_j).view(1,-1)
			# z_first = self.z.weight[i,:].view(1,-1)
			# z_second = self.z.weight[j,:].view(1,-1)

			z_elementmult_matrix[index,:] = (z_first * z_second)

			z_pair_matrix[index,:] = torch.cat((z_first,z_second),1)

		z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two
		z = torch.cat((z_pair_matrix,z_elementmult_matrix),1) @ self.weight_three
		# print(self.z(torch.LongTensor([0]).to(device))[0:5])
		return torch.sigmoid(z)

	def forward(self, x, train_edges, train_false_edges):
		edges = np.concat((train_edges, train_false_edges))
		result = self.decode(edges)
		# z2 = self.decode(train_false_edges)
		# result = torch.cat((z1,z2),0)
		return result


class NEUMF_Feature_Sample(nn.Module):
	def __init__(self,adj):
		super(NEUMF_Feature_Sample,self).__init__()
		self.layer = NEUMF_feature_SampleLayer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class NEUMF_feature_SampleLayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(NEUMF_feature_SampleLayer, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim)
		self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim*2, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def decode(self, z, edges):
		z_elementmult_matrix = torch.zeros(len(edges), self.output_dim).to(device)
		z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)

		for index, (i,j) in enumerate(edges):

			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)
			
			z_elementmult_matrix[index,:] = (z_first * z_second)

			z_pair_matrix[index,:] = torch.cat((z_first,z_second),1)

		z_pair_matrix = F.relu(z_pair_matrix @ self.weight_two)
		z = torch.cat((z_pair_matrix,z_elementmult_matrix),1) @ self.weight_three
		# print(self.z(torch.LongTensor([0]).to(device))[0:5])
		return z
	def encode(self, x):
		return torch.mm(x,self.weight)
	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		z1 = self.decode(z,train_edges)
		z2 = self.decode(z,train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)


class NEUMF_Feature_inner_product_Sample(nn.Module):
	def __init__(self,adj):
		super(NEUMF_Feature_inner_product_Sample,self).__init__()
		self.layer = NEUMF_feature_inner_product_SampleLayer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class NEUMF_feature_inner_product_SampleLayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(NEUMF_feature_inner_product_SampleLayer, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim)
		# self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def decode(self, z, edges):
		z_elementmult_matrix = torch.zeros(len(edges), self.output_dim).to(device)

		for index, (i,j) in enumerate(edges):

			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)
			
			z_elementmult_matrix[index,:] = (z_first * z_second)

		z = z_elementmult_matrix @ self.weight_three
		return z
	def encode(self, x):
		return torch.mm(x,self.weight)
	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		z1 = self.decode(z,train_edges)
		z2 = self.decode(z,train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)

class NEUMF_Feature_concat_Sample(nn.Module):
	def __init__(self,adj):
		super(NEUMF_Feature_concat_Sample,self).__init__()
		self.layer = NEUMF_feature_concat_SampleLayer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class NEUMF_feature_concat_SampleLayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(NEUMF_feature_concat_SampleLayer, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim)
		# self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim*2, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)

		for index, (i,j) in enumerate(edges):

			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)
			
			z_pair_matrix[index,:] = torch.cat((z_first, z_second),1)

		z = z_pair_matrix @ self.weight_three
		return z
	def encode(self, x):
		return torch.mm(x,self.weight)
	def forward(self, x, train_edges, train_false_edges):
		z = self.encode(x)
		z1 = self.decode(z,train_edges)
		z2 = self.decode(z,train_false_edges)
		result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)
class LGAE_Concat(nn.Module):
	def __init__(self,adj):
		super(LGAE_Concat,self).__init__()
		# self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.weight = xavier_init(args.input_dim,  args.hidden1_dim) 
		self.weight_two = xavier_init(args.hidden1_dim*2, args.hidden1_dim)
		self.weight_three = kaiming_init(args.hidden1_dim, 1)
		# self.weight_three = glorot_init(output_dim*2, 1)
		self.adj = adj
		self.input_dim = args.input_dim
		self.output_dim = args.hidden1_dim

	def encode(self, x):
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)
		return z

	def decode(self, z, edges):
		z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)
		# z_elementmult_matrix = torch.zeros(len(edges), 1).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)

			z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			# z_elementmult_matrix[index,:] = (z_first * z_second).sum()

		z_pair_matrix = F.relu(z_pair_matrix) @ self.weight_two @ self.weight_three
		# z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return torch.sigmoid(z_pair_matrix)

	def forward(self, X, train_edges, train_false_edges):
		Z = self.encode(X)
		edges = np.concatenate((train_edges,train_false_edges),0)
		A_pred = self.decode(Z, edges)
		return A_pred


class LGAE_IP_Linear(nn.Module):
	def __init__(self,adj):
		super(LGAE_IP_Linear,self).__init__()
		# self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.weight = xavier_init(args.input_dim,  args.hidden1_dim) 
		self.weight_two = xavier_init(args.hidden1_dim, 1)
		# self.weight_three = kaiming_init(args.hidden1_dim, 1)
		# self.weight_three = glorot_init(output_dim*2, 1)
		self.adj = adj
		self.input_dim = args.input_dim
		self.output_dim = args.hidden1_dim

	def encode(self, x):
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)
		return z

	def decode(self, z, edges):
		# z_pair_matrix = torch.zeros(len(edges), self.output_dim*2).to(device)
		z_elementmult_matrix = torch.zeros(len(edges), args.hidden1_dim).to(device)
		
		for index, (i,j) in enumerate(edges):
			z_first = z[i,:].view(1,-1)
			z_second = z[j,:].view(1,-1)

			# z_pair_matrix[index,:] = torch.cat((z_first , z_second),1)
			z_elementmult_matrix[index,:] = (z_first * z_second)

		z_elementmult_matrix = z_elementmult_matrix @ self.weight_two 
		# z = torch.cat((z_pair_matrix, z_elementmult_matrix),1) @ self.weight_three
		return torch.sigmoid(z_elementmult_matrix)

	def forward(self, X, train_edges, train_false_edges):
		Z = self.encode(X)
		edges = np.concatenate((train_edges,train_false_edges),0)
		A_pred = self.decode(Z, edges)
		return A_pred


class Whole_NEUMF_Sample(nn.Module):
	def __init__(self,adj):
		super(Whole_NEUMF_Sample,self).__init__()
		self.layer = Whole_NEUMF_SampleLayer(args.node_size, args.hidden1_dim, adj)

	def encode(self, X, train_edges, train_false_edges):
		A_pred = self.layer(X, train_edges, train_false_edges)
		return A_pred

	def forward(self, X, train_edges, train_false_edges):
		A_pred = self.encode(X, train_edges, train_false_edges)
		return A_pred

class Whole_NEUMF_SampleLayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(Whole_NEUMF_SampleLayer, self).__init__(**kwargs)
		self.z = normal_init(input_dim, output_dim)
		self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim*2, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	
	def forward(self, inputs, train_edges, train_false_edges):
		# x = inputs
		# x = torch.mm(x,self.weight)
		# z = torch.mm(self.adj, x)

		z1 = self.z.weight.repeat(1,self.z.weight.shape[0]).reshape(-1,self.z.weight.shape[1])
		z2 = self.z.weight.repeat(self.z.weight.shape[0],1).reshape(-1,self.z.weight.shape[1])
		z_pair = F.relu(torch.cat((z1,z2),1))
		z_elementmult = z1 * z2

		z_pair = z_pair @ self.weight_two
		z = torch.cat((z_pair, z_elementmult),1) @ self.weight_three
		z = z.reshape(args.node_size, args.node_size)
		outputs = torch.sigmoid(z)
		return outputs


class NEUMF_Feature(nn.Module):
	def __init__(self,adj):
		super(NEUMF_Feature,self).__init__()
		self.layer = NEUMF_feature_Layer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X):
		A_pred = self.layer(X)
		return A_pred

	def forward(self, X):
		A_pred = self.encode(X)
		return A_pred

class NEUMF_feature_Layer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, **kwargs):
		super(NEUMF_feature_Layer, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim)
		self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim*2, 1)

		self.adj = adj
		self.input_dim = input_dim
		self.output_dim = output_dim

	def decode(self, z):
		z1 = z.repeat(1,z.shape[0]).reshape(-1,z.shape[1])
		z2 = z.repeat(z.shape[0],1).reshape(-1,z.shape[1])
		z_pair = F.relu(torch.cat((z1,z2),1))
		z_elementmult = z1 * z2

		z_pair = z_pair @ self.weight_two
		z = torch.cat((z_pair, z_elementmult),1) @ self.weight_three
		z = z.reshape(args.node_size, args.node_size)
		outputs = torch.sigmoid(z)
		return outputs

	def encode(self, x):
		return torch.mm(x,self.weight)
	def forward(self, x):
		z = self.encode(x)
		result = self.decode(z)
		
		# result = torch.cat((z1,z2),0)
		return torch.sigmoid(result)


class WholeNLGF(nn.Module):
	def __init__(self,adj):
		super(WholeNLGF,self).__init__()
		self.layer = WholeLayer(args.input_dim, args.hidden1_dim, adj)

	def encode(self, X):
		A_pred = self.layer(X)
		return A_pred

	def forward(self, X):
		A_pred = self.encode(X)
		return A_pred

class WholeLayer(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(WholeLayer, self).__init__(**kwargs)
		self.weight = xavier_init(input_dim, output_dim) 
		self.weight_two = xavier_init(output_dim*2, output_dim)
		self.weight_three = kaiming_init(output_dim*2, 1)
		self.adj = adj
		self.activation = activation
		self.input_dim = input_dim

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		z = torch.mm(self.adj, x)

		z1 = z.repeat(1,z.shape[0]).reshape(-1,z.shape[1])
		z2 = z.repeat(z.shape[0],1).reshape(-1,z.shape[1])
		z_pair = F.relu(torch.cat((z1,z2),1))
		z_elementmult = z1 * z2

		z_pair = z_pair @ self.weight_two
		z = torch.cat((z_pair, z_elementmult),1) @ self.weight_three
		z = z.reshape(args.node_size, args.node_size)
		outputs = torch.sigmoid(z)
		return outputs
