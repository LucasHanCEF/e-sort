import torch
import torch.nn as nn
import MEAutility as MEA


class backbone(nn.Module):
    def __init__(self) -> None:
        super(backbone, self).__init__()
        self.temporal_filter1 = nn.Sequential(
            nn.Linear(60, 8),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )
        self.temporal_filter2 = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )
        self.spatial_filter1 = nn.Sequential(
            nn.Linear(16*384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # sqmea = MEA.return_mea(probe)
        # self.H = sqmea.dim[0]
        # self.W = sqmea.dim[1]

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute([0,2,1])
        # print(x.shape)
        x = self.temporal_filter1(x)
        # print(f'temp1:{x}')
        x = self.temporal_filter2(x)
        # print(f'temp2:{x}')
        # x = x.permute([0,2,1]).reshape([B, 16, self.W, self.H])
        # print(x.shape)
        x = x.reshape(B, -1)
        x = self.spatial_filter1(x)
        # print(x.shape)
        # print(f'spat1:{x}')
        # x = self.fc(x)
        return x
    
    def model_fix_para(self, mode='tf'):
        for param in self.temporal_filter1.parameters():
            param.requires_grad = False
        for param in self.temporal_filter2.parameters():
            param.requires_grad = False


class classifier(nn.Module):
    def __init__(self, num_class: int) -> None:
        super(classifier, self).__init__()
        self.num_class = num_class
        self.fc = nn.Sequential(
            nn.Linear(256, self.num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
