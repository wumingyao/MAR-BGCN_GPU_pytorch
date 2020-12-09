import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from lib.metrics import masked_mape_np
from scipy.sparse.linalg import eigs
import math
import queue
from time import time
from lib.metrics import masked_mae_torch

def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float64)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float64)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    identity = np.identity(W.shape[0])
    return ((2 * L) / lambda_max - identity)


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


def load_graphdata(graph_signal_matrix_filename, len_input, num_for_predict, batch_size, DEVICE, shuffle=True):
    '''
       数据准备
       将x,y都处理成归一化到[-1,1]之前的数据;
       每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
       注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
       返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
       :param graph_signal_matrix_filename: str
       :param num_for_predict: int
       :param num_for_predict: int
       :param batch_size: int
       :param DEVICE:
       :return:
       Train,Val,Test three DataLoaders,each dataLoader contain,ie:
       train_loader:(B,T',N,F)
       train_target_tensor:((B,T,N,F))
       two keys: mean and std
       '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    filename = os.path.join(dirpath, file + '_h' + str(len_input) + '_p' + str(num_for_predict)) + '_mracgn.npz'
    print('load file:', filename)
    file_data = np.load(filename)
    train_x = file_data['train_x']  # (sample,T',N,F)
    train_accident = file_data['train_accident']
    train_x_sub = file_data['train_x_sub']
    train_adj_sub = file_data['train_adj_sub']
    train_target = file_data['train_target']  # (sample,T,N,F)

    val_x = file_data['val_x']
    val_accident = file_data['val_accident']
    val_x_sub = file_data['val_x_sub']
    val_adj_sub = file_data['val_adj_sub']
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_accident = file_data['test_accident']
    test_x_sub = file_data['test_x_sub']
    test_adj_sub = file_data['test_adj_sub']
    test_target = file_data['test_target']

    mean = file_data['mean']
    std = file_data['std']

    # ------- train_loader -------

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor)
    # train_accident_tensor = torch.from_numpy(train_accident).type(torch.FloatTensor).to(DEVICE)
    # train_x_sub_tensor = torch.from_numpy(train_x_sub).type(torch.FloatTensor).to(DEVICE)
    # train_adj_sub_tensor = torch.from_numpy(train_adj_sub).type(torch.FloatTensor).to(DEVICE)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)

    # train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_accident_tensor, train_x_sub_tensor,
    #                                                train_adj_sub_tensor,
    #                                                train_target_tensor)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor)
    # val_accident_tensor = torch.from_numpy(val_accident).type(torch.FloatTensor).to(DEVICE)
    # val_x_sub_tensor = torch.from_numpy(val_x_sub).type(torch.FloatTensor).to(DEVICE)
    # val_adj_sub_tensor = torch.from_numpy(val_adj_sub).type(torch.FloatTensor).to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor)

    # val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_accident_tensor, val_x_sub_tensor,
    #                                              val_adj_sub_tensor, val_target_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor)
    # test_accident_tensor = torch.from_numpy(test_accident).type(torch.FloatTensor).to(DEVICE)
    # test_x_sub_tensor = torch.from_numpy(test_x_sub).type(torch.FloatTensor).to(DEVICE)
    # test_adj_sub_tensor = torch.from_numpy(test_adj_sub).type(torch.FloatTensor).to(DEVICE)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)

    # test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_accident_tensor, test_x_sub_tensor,
    #                                               test_adj_sub_tensor,
    #                                               test_target_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_dataset, val_dataset, test_dataset, mean, std


def compute_val_loss_parallel(net, val_loader_node, mean_node, std_node, sw,
                              epoch, batch_size, DEVICE, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net = net.train(False)  # ensure dropout layers are in evaluation mode
    start_time = time()
    with torch.no_grad():

        # val_loader_length = len(val_loader_node)  # nb of batch

        tmp = []  # 记录了所有batch的loss
        batch_val = math.ceil(len(val_loader_node) / batch_size)
        for batch_index in range(batch_val):
            if batch_index == batch_val - 1:
                encoder_inputs_node, labels_node = val_loader_node[
                                                   batch_index * batch_size:]
            else:
                encoder_inputs_node, labels_node = val_loader_node[
                                                   batch_index * batch_size:(batch_index + 1) * batch_size]
            # out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node)
            # out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node)
            encoder_inputs_node = encoder_inputs_node.to(DEVICE)
            labels_node = labels_node.to(DEVICE)
            out_node = net(encoder_inputs_node)
            # out_node = net(encoder_inputs_node, encoder_inputs_edge)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            loss = masked_mae_torch(out_node, labels_node)
            # loss = loss_node
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f, time: %.2fs' % (
                    batch_index, batch_val, loss.item(), time() - start_time))
                start_time = time()
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
        print('validation epoch %s, loss: %.2f' % (epoch, validation_loss))
    return validation_loss


def compute_val_loss(net, val_loader_node, mean_node, std_node, criterion, sw,
                     epoch, batch_size, adj_sub, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net = net.train(False)  # ensure dropout layers are in evaluation mode
    start_time = time()
    with torch.no_grad():

        # val_loader_length = len(val_loader_node)  # nb of batch

        tmp = []  # 记录了所有batch的loss
        batch_val = math.ceil(len(val_loader_node) / batch_size)
        for batch_index in range(batch_val):
            if batch_index == batch_val - 1:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = val_loader_node[
                                                                                                                         batch_index * batch_size:]
            else:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = val_loader_node[
                                                                                                                         batch_index * batch_size:(
                                                                                                                                                          batch_index + 1) * batch_size]
            out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, adj_sub,
                           encoder_adj_sub_node)
            # out_node = net(encoder_inputs_node, encoder_inputs_edge)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            loss = masked_mae_torch(out_node, labels_node)
            # loss = loss_node
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f, time: %.2fs' % (
                    batch_index, batch_val, loss.item(), time() - start_time))
                start_time = time()
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
        print('validation epoch %s, loss: %.2f' % (epoch, validation_loss))
    return validation_loss


def setup_graph(net, test_loader_node, test_loader_edge, batch_size):
    with torch.no_grad():
        net = net.eval()
        batch_test = math.ceil(len(test_loader_node) / batch_size)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                encoder_inputs_node, labels_node = test_loader_node[
                                                   batch_index * batch_size:]
                encoder_inputs_edge, labels_edge = test_loader_edge[
                                                   batch_index * batch_size:]
            else:
                encoder_inputs_node, labels_node = test_loader_node[
                                                   batch_index * batch_size:(batch_index + 1) * batch_size]
                encoder_inputs_edge, labels_edge = test_loader_edge[
                                                   batch_index * batch_size:(batch_index + 1) * batch_size]

            out_node, out_edge = net(encoder_inputs_node, encoder_inputs_edge)
            break
        return net


def load_mode(net, test_loader_node, test_loader_edge, batch_size, params_filename):
    net = setup_graph(net, test_loader_node, test_loader_edge, batch_size)
    net.load_state_dict(torch.load(params_filename))
    return net


def evaluate_on_test_parallel(net, best_epoch, params_path, test_loader_node, mean_node, std_node, batch_size, DEVICE):
    '''
    :param global_step:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param type:
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % best_epoch)
    print('load weight from:', params_filename)
    # 加载模型
    net.load_state_dict(torch.load(params_filename))
    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)
    net.eval()
    with torch.no_grad():
        prediction_node = []
        prediction_edge = []
        true_node = []
        true_edge = []
        batch_test = math.ceil(len(test_loader_node) / batch_size)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                encoder_inputs_node, labels_node = test_loader_node[
                                                   batch_index * batch_size:]
            else:
                encoder_inputs_node, labels_node = test_loader_node[
                                                   batch_index * batch_size:(
                                                                                    batch_index + 1) * batch_size]

            # out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node,
            #                encoder_adj_sub_node)
            encoder_inputs_node = encoder_inputs_node.to(DEVICE)
            labels_node = labels_node.to(DEVICE)
            out_node = net(encoder_inputs_node)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            prediction_node.extend(out_node.cpu().numpy()[:, -1].flatten().tolist())
            true_node.extend(labels_node.cpu().numpy()[:, -1].flatten().tolist())

        # prediction_node = np.array(prediction_node)
        # prediction_node = np.maximum(prediction_node, 0)
        # prediction_edge = np.array(prediction_edge)
        # prediction_edge = np.maximum(prediction_edge, 0)
        # print('预测')
        # print(np.array(prediction_node))
        # print('真实')
        # print(np.array(true_node))
        mae_node = mean_absolute_error(np.array(true_node), np.array(prediction_node))
        mse_node = mean_squared_error(np.array(true_node), np.array(prediction_node))
        rmse_node = mean_squared_error(np.array(true_node), np.array(prediction_node)) ** 0.5
        mape_node = masked_mape_np(np.array(true_node), np.array(prediction_node), 0)

        print('mae_node: %.2f' % (mae_node))
        print('mse_node: %.2f' % (mse_node))
        print('rmse_node: %.2f' % (rmse_node))
        print('mape_node: %.2f' % (mape_node))


def evaluate_on_test(net, best_epoch, params_path, test_loader_node, mean_node, std_node, batch_size, adj_sub):
    '''
    :param global_step:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param type:
    :return:
    '''
    params_filename = os.path.join(params_path, 'epoch_%s.params' % best_epoch)
    print('load weight from:', params_filename)
    # 加载模型
    net.load_state_dict(torch.load(params_filename))
    # print(net)
    # for name, param in net.named_parameters():
    #     print(name, param)
    net.eval()
    with torch.no_grad():
        prediction_node = []
        prediction_edge = []
        true_node = []
        true_edge = []
        batch_test = math.ceil(len(test_loader_node) / batch_size)
        for batch_index in range(batch_test):
            if batch_index == batch_test - 1:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = test_loader_node[
                                                                                                                         batch_index * batch_size:]
            else:
                encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, encoder_adj_sub_node, labels_node = test_loader_node[
                                                                                                                         batch_index * batch_size:(
                                                                                                                                                          batch_index + 1) * batch_size]

            out_node = net(encoder_inputs_node, encoder_accident_node, encoder_inputs_sub_node, adj_sub,
                           encoder_adj_sub_node)
            out_node = out_node * mean_node[0, 0, 0, 0] + std_node[0, 0, 0, 0]
            prediction_node.extend(out_node.cpu().numpy().flatten().tolist())
            true_node.extend(labels_node.cpu().numpy().flatten().tolist())

        # prediction_node = np.array(prediction_node)
        # prediction_node = np.maximum(prediction_node, 0)
        # prediction_edge = np.array(prediction_edge)
        # prediction_edge = np.maximum(prediction_edge, 0)
        # print('预测')
        # print(np.array(prediction_node))
        # print('真实')
        # print(np.array(true_node))
        mae_node = mean_absolute_error(np.array(true_node), np.array(prediction_node))
        mse_node = mean_squared_error(np.array(true_node), np.array(prediction_node))
        rmse_node = mean_squared_error(np.array(true_node), np.array(prediction_node)) ** 0.5
        mape_node = masked_mape_np(np.array(true_node), np.array(prediction_node), 0)

        print('mae_node: %.2f' % (mae_node))
        print('mse_node: %.2f' % (mse_node))
        print('rmse_node: %.2f' % (rmse_node))
        print('mape_node: %.2f' % (mape_node))


def predict_and_save_results(net, test_loader_node, test_loader_edge, global_step):
    '''
    :param net:
    :param data_loader:
    :param test_loader_node:
    :param test_loader_edge:
    :param mean_node:
    :param std_node:
    :param mean_edge:
    :param std_edge:
    :param global_step:
    :param params_path:
    :param type:
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        prediction_node = []
        prediction_edge = []
        target_node = []
        target_edge = []
        loader_length = len(test_loader_node)

        for batch_index in range(loader_length):

            encoder_inputs_node, labels_node = test_loader_node[batch_index]
            encoder_inputs_edge, labels_edge = test_loader_edge[batch_index]

            out_node, out_edge = net(encoder_inputs_node, encoder_inputs_edge)

            prediction_node.append(out_node)
            target_node.append(labels_node)
            prediction_edge.append(out_edge)
            target_edge.append(labels_edge)

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        prediction_node = np.concatenate(prediction_node, 0)
        target_node = np.concatenate(target_node, 0)
        prediction_edge = np.concatenate(prediction_edge, 0)
        target_edge = np.concatenate(target_edge, 0)

        # 计算误差
        excel_list = []
        prediction_length = prediction_node.shape[1]

        # 计算每个步长的误差
        for i in range(prediction_length):
            assert prediction_node.shape[0] == target_node.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            mae = mean_absolute_error(target_node[:, i, :, :], prediction_node[:, i, :, :])
            rmse = mean_squared_error(target_node[:, i, :, :], prediction_node[:, i, :, :]) ** 0.5
            mape = masked_mape_np(target_node[:, i, :, :], prediction_node[:, i, :, :], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(target_node, prediction_node)
        rmse = mean_squared_error(target_node, prediction_node) ** 0.5
        mape = masked_mape_np(target_node, prediction_node, 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


def bfs(adj, start):
    visited = set()
    visited.add(start)
    q = queue.Queue()
    q.put(start)  # 把起始点放入队列
    while not q.empty():
        u = q.get()
        # print(u)
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.put(v)
    return list(visited)


def search_subgraph(sensor, adj, N_sub):
    '''
    :param sensor: sensor在idlist中的序列号
    :param adj: 全局adj
    :return: list,len=N_sub
    '''
    res = bfs(adj, sensor)
    if len(res) > N_sub:
        res = res[:N_sub]
    sid = 0
    while len(res) < N_sub:
        res.append(sid)
        res = list(set(res))
        sid += 1
    res.sort()
    return res
