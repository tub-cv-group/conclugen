
# CODE From
# https://github.com/FelixOpolka/STGCN-PyTorch/blob/master/stgcn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.con_clu.attention import PAM_Module


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, att_dim):
        super(AttentionBlock, self).__init__()
        #self.lstm =  nn.LSTM(in_channels,out_channels)
        out_channels = 4
        self.temporal = TimeBlock(in_channels, out_channels)
        self.fc = nn.Linear((att_dim*out_channels)*2, 1)
        self.indexes = None

    def forward(self, X, A):
        if self.indexes is None:
            self.indexes = torch.nonzero(A)

        t = self.temporal(X)
        b_size = t.shape[0]

        att_mat = torch.zeros(b_size, A.shape[0], A.shape[1]).to(
            A.device)  # matrix [batch, num_nodes, num_nodes]

        for idx in self.indexes:
            i, j = idx[0], idx[1]
            concat = torch.cat(
                (t[:, i].flatten(start_dim=1), t[:, j].flatten(start_dim=1)), 1)
            att_mat[:, i, j] = F.leaky_relu(self.fc(concat).squeeze(1))

        att_mat = F.softmax(att_mat, dim=2) * A
        return att_mat


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, att_dim, attention=True):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        #print(f"in_channels {in_channels} spatial_channels {spatial_channels} out_channels {out_channels} num_nodes {num_nodes}")
        #self.attention_block = AttentionBlock(in_channels,att_dim)

        self.attention = attention
        if self.attention:
            self.attention_block_time = PAM_Module(att_dim)
            self.attention_block_space = PAM_Module(51)

        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        #print(f"X {X.shape} ")
        t = self.temporal1(X)

        #[batch, num_nodes, num_nodes]
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])

        # attenion using CAM or PAM
        if self.attention:
            time_lsf = lfs.permute(0, 2, 1, 3).contiguous()
            time_lsf = self.attention_block_time(time_lsf)
            time_lsf = time_lsf.permute(0, 2, 1, 3).contiguous()
            space_lsf = self.attention_block_space(lfs)
            lfs = time_lsf + space_lsf

        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # print(f"{t2.shape}")

        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


class MiniBlock(nn.Module):
    """
    Mini STGCN block for each face section.
    """

    def __init__(self, num_nodes, num_features=64, att_dim=88, attenion=True):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(MiniBlock, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, att_dim=att_dim, attention=attenion)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes, att_dim=att_dim-4, attention=attenion)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):

        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        return out3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(
        self,
        num_nodes,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        num_classes=8,
        edge_weight=False,
        contrastive=False,
        separate_graph=False,
        attention=False
    ):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()

        self.contrastive = contrastive
        self.separate_graph = separate_graph
        if self.separate_graph:

            # for simplicity the array are indexed as 0 :r_eye 1:l_eye 2:nose 3:mouth
            self.graph_kp = [[0, 11], [12, 12+11], [23, 23+9], [31, 51]]
            nodes = [11, 11, 9, 20]

            self.eye_r_block = MiniBlock(
                nodes[0], num_features=num_features, attention=attention)
            self.eye_l_block = MiniBlock(
                nodes[1], num_features=num_features, attention=attention)
            self.nose_block = MiniBlock(
                nodes[2], num_features=num_features, attention=attention)
            self.mouth_block = MiniBlock(
                nodes[3], num_features=num_features, attention=attention)

            self.global_block = MiniBlock(
                4, num_features=64, att_dim=78, attention=attention)
            self.fully = nn.Linear(
                (num_timesteps_input - 20) * 64, num_timesteps_output)

            if not self.contrastive:
                self.fc_out = nn.Linear(4*num_timesteps_output, num_classes)
            else:
                self.fc_out = torch.nn.Sequential(torch.nn.Linear(
                    4*num_timesteps_output, 512), torch.nn.ReLU(), torch.nn.Linear(512, num_classes))
            #self.fc_out = nn.Linear(num_nodes*num_timesteps_output,num_classes)

        else:
            self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                     spatial_channels=16, num_nodes=num_nodes, att_dim=88, attention=attention)
            self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                     spatial_channels=16, num_nodes=num_nodes, att_dim=84, attention=attention)
            self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
            self.fully = nn.Linear(
                (num_timesteps_input - 2 * 5) * 64, num_timesteps_output)

            self.dropout = nn.Dropout(p=0.1)

            if not self.contrastive:
                self.fc_out = nn.Linear(
                    num_nodes*num_timesteps_output, num_classes)
            else:
                self.fc_out = torch.nn.Sequential(torch.nn.Linear(
                    num_nodes*num_timesteps_output, 512), torch.nn.ReLU(), torch.nn.Linear(512, num_classes))
            #self.fc_out = nn.Linear(num_nodes*num_timesteps_output,num_classes)

        if edge_weight:
            m = torch.Tensor(get_normalized_adj(np.ones((51, 51))))
            self.edge_importance1 = nn.Parameter(m)
            m = torch.Tensor(get_normalized_adj(np.ones((51, 51))))
            self.edge_importance2 = nn.Parameter(m)
        else:
            self.edge_importance1 = 1  # torch.ones((51,51))
            self.edge_importance2 = 1  # torch.ones((51,51))
        self.edge_weight = edge_weight
        self.mask = torch.zeros(51, 51)

    def forward(self, A_hat, X, eval=False, augmented=False):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """

        if self.separate_graph:
            X = X.permute(0, 2, 1, 3)
            eye_r = self.eye_r_block(X[:, self.graph_kp[0][0]:self.graph_kp[0][1]], A_hat[self.graph_kp[0]
                                     [0]:self.graph_kp[0][1], self.graph_kp[0][0]:self.graph_kp[0][1]]).mean(axis=1).unsqueeze(1)
            #print(f"eye_r.shape {eye_r.shape}")
            eye_l = self.eye_l_block(X[:, self.graph_kp[1][0]:self.graph_kp[1][1]], A_hat[self.graph_kp[1]
                                     [0]:self.graph_kp[1][1], self.graph_kp[1][0]:self.graph_kp[1][1]]).mean(axis=1).unsqueeze(1)
            #print(f"eye_l.shape {eye_l.shape}")
            nose = self.nose_block(X[:, self.graph_kp[2][0]:self.graph_kp[2][1]], A_hat[self.graph_kp[2]
                                   [0]:self.graph_kp[2][1], self.graph_kp[2][0]:self.graph_kp[2][1]]).mean(axis=1).unsqueeze(1)
            #print(f"nose.shape {nose.shape}")
            mouth = self.mouth_block(X[:, self.graph_kp[3][0]:self.graph_kp[3][1]], A_hat[self.graph_kp[3]
                                     [0]:self.graph_kp[3][1], self.graph_kp[3][0]:self.graph_kp[3][1]]).mean(axis=1).unsqueeze(1)
            #print(f"mouth.shape {mouth.shape} ")
            global_input = torch.cat((eye_r, eye_l, nose, mouth), 1)
            #print(f"global_input.shape {global_input.shape}")
            mini_A = torch.ones((4, 4)).to(X.device)
            #print(f"mini_a {mini_A.shape}")
            global_out = self.global_block(global_input, mini_A)
            #print(f"global_out.shape {global_out.shape}")
            #print(global_out.reshape((global_out.shape[0], global_out.shape[1], -1)).shape)
            out4 = self.fully(global_out.reshape(
                (global_out.shape[0], global_out.shape[1], -1)))
            # print(out4.shape)
        else:

            if self.edge_weight:
                if self.mask.sum() == 0:
                    self.mask = self.mask.to(A_hat.device)
                    self.mask[torch.where(A_hat == 0)] = 1
                edge1 = self.edge_importance1 * self.mask
                edge2 = self.edge_importance2 * self.mask
            else:
                edge1 = 0
                edge2 = 0

            out1 = self.block1(X.permute(0, 2, 1, 3), A_hat + edge1)
            out2 = self.block2(out1, A_hat + edge2)
            out3 = self.dropout(self.last_temporal(out2))
            out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        if not self.contrastive:
            out4 = self.fc_out(out4.flatten(start_dim=1))
            if eval:
                out4 = F.softmax(out4)
            return out4
        else:
            outputs = self.fc_out(out4.flatten(start_dim=1))
            return torch.nn.functional.normalize(outputs), out4.flatten(start_dim=1)
