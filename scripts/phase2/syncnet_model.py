"""SyncNet network definition, vendored verbatim from joonson/syncnet_python.

Only the model is vendored: syncnet.py replaces SyncNetInstance.evaluate with a
streaming version that never writes frames to disk. Weights (syncnet_v2.model) are
downloaded on first use by syncnet.py.

Original: https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py

Copyright (c) 2016-present Joon Son Chung.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn

class S(nn.Module):
    def __init__(self, num_layers_in_fc_layers = 1024):
        super().__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):

        mid = self.netcnnaud(x)  # N x ch x 24 x M
        mid = mid.view((mid.size(0), -1))  # N x (ch x 24)
        out = self.netfcaud(mid)

        return out

    def forward_lip(self, x):

        mid = self.netcnnlip(x)
        mid = mid.view((mid.size(0), -1))  # N x (ch x 24)
        out = self.netfclip(mid)

        return out

    def forward_lipfeat(self, x):

        mid = self.netcnnlip(x)
        out = mid.view((mid.size(0), -1))  # N x (ch x 24)

        return out
