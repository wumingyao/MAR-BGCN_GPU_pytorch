# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
import torch.optim as optim


class GCN(nn.Module):
    '''
    K-order chebyshev graph convolution
    计算khop的GCN
    '''

    def __init__(self, adj, dim_in, dim_out, order_K, device, in_drop=0.0, gcn_drop=0.0, residual=False):
        '''
        :param adj:邻接矩阵
        :param K: int,num of aggregated neighbors
        :param dim_in: int, num of channels in the input sequence
        :param dim_out: int, num of channels in the output sequence
        '''
        super(GCN, self).__init__()
        self.DEVICE = device
        self.order_K = order_K
        self.adj = adj
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(dim_in, dim_out))
             for _ in range(order_K)])
        self.weights = nn.Parameter(torch.FloatTensor(size=(dim_out, dim_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(dim_out,)))
        self._in_drop = in_drop
        self._gcn_drop = gcn_drop
        self._residual = residual
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, state=None, M=None):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size,N, dim_in)
        :return: (batch_size,N, dim_out)
        '''
        batch_size, num_of_vertices, in_channels = x.shape
        output = torch.zeros(batch_size, num_of_vertices, self.dim_out).to(self.DEVICE)  # (batch_size,N, dim_out)
        L_tilde = scaled_Laplacian(self.adj)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in
                            cheb_polynomial(L_tilde, self.order_K)]
        if state is not None:
            s = torch.einsum('ij,jkm->ikm', M, state.permute(1, 0, 2)).permute(1, 0, 2)
            x = torch.cat((x, s), dim=-1)
        x0 = x
        if self._in_drop != 0:
            x = torch.dropout(x, 1.0 - self._in_drop, train=True)
        # k-order展开
        for k in range(self.order_K):
            # chebyshev多项式
            output = output + x.permute(0, 2, 1).matmul(cheb_polynomials[k].to(self.DEVICE)).permute(0, 2, 1).matmul(self.Theta[k])
        output = torch.matmul(output, self.weights)
        output = output + self.biases
        res = F.relu(output)
        if self._gcn_drop != 0.0:
            res = torch.dropout(res, 1.0 - self._gcn_drop, train=True)
        if self._residual:
            x0 = self.linear(x0)
            res = res + x0
        return res  # (batch_size,N, dim_out)
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import dgl.function as fn
# from dgl import DGLGraph
#
# class GCN(nn.Module):
#     def __init__(self, G, dim_in, dim_h, dim_z, n_class, dropout):
#         super(GCN, self).__init__()
#         self.G = G
#         self.dim_z = dim_z
#         self.layer0 = GCN_layer(G, dim_in, dim_h, dropout)
#         self.layer1 = GCN_layer(G, dim_h, dim_z, dropout)
#         self.layer2 = GCN_layer(G, dim_z, n_class, dropout, act=False)
#
#     def forward(self, features, norm):
#         h = self.layer0(features, norm)
#         h = self.layer1(h, norm)
#         x = self.layer2(h, norm)
#         return x
#
# class GCN_layer(nn.Module):
#     def __init__(self, G, dim_in, dim_out, dropout, act=True):
#         super(GCN_layer, self).__init__()
#         self.G = G
#         self.act = act
#         self.dropout = dropout
#         self.weight = self.glorot_init(dim_in, dim_out)
#         self.linear = nn.Linear(dim_in, dim_out, bias=False)
#         if self.dropout:
#             self.dropout = nn.Dropout(p=dropout)
#
#     def glorot_init(self, input_dim, output_dim):
#         init_range = np.sqrt(6.0/(input_dim + output_dim))
#         initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
#         return nn.Parameter(initial)
#
#     def forward(self, h, norm):
#         if self.dropout:
#             h = self.dropout(h)
#         h = h @ self.weight
#         self.G.ndata['h'] = h * norm
#         self.G.update_all(fn.copy_src(src='h', out='m'),
#                           fn.sum(msg='m', out='h'))
#         h = self.G.ndata.pop('h') * norm
#         if self.act:
#             h = F.relu(h)
#         return h
#
#
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
#
# from models import GCN
# from dataloader import DataLoader
# from utils import train_model
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='GCN')
#     parser.add_argument('--cuda', action='store_true', help='declare if running on GPU')
#     parser.add_argument('--emb_size', type=int, default=64, help='dimension of the latent space')
#     parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
#     parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
#     parser.add_argument('--seed', type=int, default=7, help='random seed, -1 for not fixing it')
#     parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
#     parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
#     parser.add_argument('--dataset', type=str, default='cora')
#     args = parser.parse_args()
#     return args
#
#
# def main(args):
#     # config device
#     args.device = torch.device('cuda' if args.cuda else 'cpu')
#     # fix random seeds
#     if args.seed > 0:
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#
#     dl = DataLoader(args)
#
#     model = GCN(dl.G, dl.features.size(1), args.hidden_size, args.emb_size, dl.n_class, args.dropout)
#     model.to(args.device)
#     model = train_model(args, dl, model)
#
#
# if __name__ == "__main__":
#     args = get_args()
#     main(args)
#
# g = dgl.graph(([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 1, 2, 3, 4, 0]))
# # g.ndata['h'] = torch.ones(5, 1)
# # g = dgl.khop_graph(g, 3)
#
# gl = GCNLayer(1, 5)
# res = gl(g, torch.ones(5, 1))
# pass
