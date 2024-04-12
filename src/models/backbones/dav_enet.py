import math
import shutil
import subprocess

import torch
import torch.nn as nn
import numpy as np
import scipy.signal
import gdown
import os
import tarfile


def conv1x9(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,9), stride=stride, padding=(0,4), bias=False)


def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), stride=stride, padding=(0,pad_amt), bias=bias)


class SpeechBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(SpeechBasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResDavenet(nn.Module):

    # Disclaimer: Model taken from here: https://github.com/wnhsu/ResDAVEnet-VQ

    def __init__(
        self,
        feat_dim=40,
        block=SpeechBasicBlock,
        layers=[2, 2, 2, 2],
        layer_widths=[128, 256, 256, 512, 1024],
        convsize=9,
        pretrained=True
    ):
        super(ResDavenet, self).__init__()
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(self.feat_dim, 1), stride=1, padding=(0,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], width=convsize, stride=2)
        self.layer3 = self._make_layer(block, layer_widths[3], layers[2], width=convsize, stride=2)
        self.layer4 = self._make_layer(block, layer_widths[4], layers[3], width=convsize, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            torch_home = os.environ['TORCH_HOME']
            out_dir = os.path.join(torch_home, 'checkpoints')
            checkpoint_path = os.path.join(out_dir, 'davenet.pth')
            if not os.path.exists(checkpoint_path):
                os.makedirs(out_dir, exist_ok=True)
                download_path = os.path.join(out_dir, 'davenet.tar.gz')
                # Pretrained DaveNet
                url = f"curl 'https://drive.usercontent.google.com/download?id=1J-tw3eg3R5e9k0vIfQaBVKiaHyOJUcIB&export=download&authuser=0&confirm=t&uuid=4ef8d4db-15e7-44f0-877a-7107e9943103&at=APZUnTVkld-DkyowUfde8zaky7pQ%3A1711549475788' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Connection: keep-alive' -H 'Referer: https://drive.usercontent.google.com/' -H 'Cookie: AEC=Ae3NU9MZnz9immtUNErvvW-4fY2Pb94bz3nQuKU5Tan2vaJf9K9jlOUa9A; CONSENT=PENDING+814; SOCS=CAISHAgCEhJnd3NfMjAyNDAxMTAtMF9SQzEaAmRlIAEaBgiA65GtBg; NID=512=sjJzEitoK0QnEO4COgwRPt1bRcK-MJXmVz9MaqJjhpcrO_X9OVN_tMmpwOifUfiFNnV2hCRbON4xyMoIdL_fYLt_UYUuFWW0Twn9PybS7wBp9rerPkrlPCNsr1X3045u92M-PI2jJG19EqslaV7mIhW0JwGXHHUJKC16Sn-cHDNc9UHI3z-DdrPi-hP93KrbcaUZzTwuj0Q2RSaVkTA0UCyTCFdiTsQr4jfldFly2_rz0FgM8otr4rElMauAVm5RjcDxBzI6TU5jIIfaYAm6qk2QPuKjbYnpRwG7dksDdUoKfivloDndeBUwvA2d1OU7SWEHkK4zuaIYrRC7jHqC4YEh1uBIjeRu3IrBsfVDTDUAW4ksAfiiT5iBjzVYCjVDTyZC3Dx84w; SID=g.a000hgg6IeX-MSBBujhlyvGn7DtXYgp50QP5hUX0kIPMiIugfSG0wSq81kEd3APiX_dAx9drBwACgYKAfASAQASFQHGX2Miv8vaRl5Axb__SQpD8VZDrBoVAUF8yKqJUL9NXUyZ6PDgA4aBhT0R0076; __Secure-1PSID=g.a000hgg6IeX-MSBBujhlyvGn7DtXYgp50QP5hUX0kIPMiIugfSG00kTgcVxAWqOdHGeBfiUMHgACgYKAV8SAQASFQHGX2MiGTMRrQFeuQz9W0L_d4w7-BoVAUF8yKoFxOARNwKLGF55r6LPm2xr0076; __Secure-3PSID=g.a000hgg6IeX-MSBBujhlyvGn7DtXYgp50QP5hUX0kIPMiIugfSG0z3H6fK_IQ6-QtZrDTkjWagACgYKAUwSAQASFQHGX2MiS0QhEFB8kSP6nGNK-s4TXhoVAUF8yKoyR0aEFPh21ehBFU_-ChUU0076; HSID=ASDnzbLNdrocG59Gs; SSID=AkOrdw4T9-sZq2Pf4; APISID=zl5eQTSeedWu6Rm8/Ai4ZChH3eLLy2OkMK; SAPISID=w3a9klvfOSlIFnNF/AJFKBF-Nc7cNgwz2j; __Secure-1PAPISID=w3a9klvfOSlIFnNF/AJFKBF-Nc7cNgwz2j; __Secure-3PAPISID=w3a9klvfOSlIFnNF/AJFKBF-Nc7cNgwz2j; SIDCC=AKEyXzVRE38GbzDt0Ej2AsEPu-h1uj8GFlkWPBYEGmAJn14wCIHoLqV5_HWqK3TaiTi9dPvyg4k; __Secure-1PSIDCC=AKEyXzUBoDHmQQP9Yzg2mK7mnGETD4nuUV5Q8QisxogJHASyHrKsc-WsbZq7Zed1aY3ikhBw9tY; __Secure-3PSIDCC=AKEyXzVB1UuNJeZBDFs-eQnw3bV_muJYqE-U88wFRWew2vlvngjAAOP7vK8ZJ1AbioGYfOPnYq0; __Secure-1PSIDTS=sidts-CjEB7F1E_DUDZRd7AvQL5smM4rLRI8gCs6CwH6OxCBDi_DqivpPjxC8cjhPsKkZJwACZEAA; __Secure-3PSIDTS=sidts-CjEB7F1E_DUDZRd7AvQL5smM4rLRI8gCs6CwH6OxCBDi_DqivpPjxC8cjhPsKkZJwACZEAA; 1P_JAR=2024-3-20-12' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: cross-site' -H 'Sec-Fetch-User: ?1' --output {download_path}"
                subprocess.run(url, shell=True)
                compressed_file = tarfile.open(download_path, 'r')
                compressed_artifact_path = os.path.join(
                    'RDVQ_00000', 'models', 'best_audio_model.pth')
                compressed_file.extract(
                    os.path.join(compressed_artifact_path),
                    out_dir)
                shutil.move(
                    os.path.join(out_dir, compressed_artifact_path),
                    out_dir)
                os.rename(
                    os.path.join(out_dir, 'best_audio_model.pth'),
                    checkpoint_path)
            self.load_state_dict(torch.load(checkpoint_path))
            

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze(2)
        return x