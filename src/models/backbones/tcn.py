# Code from
# https://github.com/VeroJulianaSchmalz/E2E-Sentence-Classification-on-Fluent-Speech-Commands/blob/main/models.py

import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNNet(nn.Module):
    def __init__(self, n_frames=200, n_feats=40, kernel=5, max_pooling=2):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=kernel)
        conv1_outsize = (20, n_feats - kernel + 1, n_frames - kernel + 1)

        self.conv2 = nn.Conv2d(20, 20, kernel_size=kernel)
        conv2_outsize = (conv1_outsize[0], int(conv1_outsize[1] / max_pooling - kernel + 1),
                         int(conv1_outsize[2] / max_pooling - kernel + 1))

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(
            int(conv2_outsize[0] * (conv2_outsize[1] / max_pooling) * (conv2_outsize[2] / max_pooling)), 500)
        self.fc2 = nn.Linear(500, 248)

        self.max_pooling = max_pooling

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = F.relu(F.max_pool2d(x, self.max_pooling))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), self.max_pooling))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return torch.sigmoid(x)
        return x


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


EPS = 1e-8


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].
    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from
            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm
    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, ):
        super(Conv1DBlock, self).__init__()
        conv_norm = GlobLN
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class TCN(nn.Module):

    # n blocks --> receptive field increases , n_repeats increases capacity mostly
    def __init__(
        self, out_features, in_chan=128, n_src=1, n_blocks=5, n_repeats=2,
        bn_chan=64, hid_chan=128, kernel_size=3, cut_out=False
    ):
        """
        out_chan should be set to the number of classes
        """
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_features = out_features if out_features else in_chan
        self.out_features = out_features
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.cut_out = cut_out

        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.1)

        for r in range(n_repeats):  # ripetizioni 2
            for x in range(n_blocks):  # 5 layers convoluzionali
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan,
                                            kernel_size, padding=padding,
                                            dilation=2 ** x))

        self.fc = nn.Linear(bn_chan, n_src * out_features)
        self.out = nn.Sequential(nn.PReLU(), self.fc)

    def classification_layer(self):
        return self.fc

    def set_classification_layer(self, classification_layer):
        self.fc= classification_layer

    # Get activation function.
    def forward(self, input,**kwargs):
        input = input.permute(0, 2, 1)
        output = self.dropout(self.bottleneck(input))
        for i in range(len(self.TCN)):
            residual = self.dropout(self.TCN[i](output))
            output = output + residual

        # provare max pool2D su ouput seguito de reshape .view(-1,1)
        if self.cut_out:
            return output.mean(1)
        logits = self.out(output.mean(-1))
        #output=nn.PReLU(output)
        #logits = self.classification_layer(output.mean(-1))
        # output = F.max_pool2d(output, 4).view(output.size(0),-1)
        # logits = self.out(output)
        return logits
