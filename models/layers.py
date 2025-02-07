import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self,
                in_channels,
                num_residual_hiddens):
        super(Residual, self).__init__()
        
        num_residual_hiddens = int(in_channels*num_residual_hiddens)
        
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=in_channels,
                    out_channels=num_residual_hiddens,
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=num_residual_hiddens,
                    out_channels=in_channels,
                    kernel_size=1, stride=1, bias=False
                    )
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class ResidualStack(nn.Module):
    def __init__(self,
                in_channels,
                num_residual_layers,
                num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([
            Residual(in_channels, num_residual_hiddens) for _ in range(self.num_residual_layers)
        ])
    
    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
            return F.relu(x)
        
class Encoder(nn.Module):
    def __init__(self,
                in_length,
                in_channels,
                num_hiddens,
                num_residual_layers,
                num_residual_hiddens):
        super(Encoder, self).__init__()
        
        current_ch = in_channels
        conv_layers = nn.ModuleList()
        next_ch = None
        current_length = in_length
        next_length = None
        while True:
            if next_ch is None:
                next_ch = current_ch*4
                next_length = current_length//2
            if (num_hiddens < next_ch) or (next_length < 32):
                break
            
            #Conv1d For Squeezing
            conv_layers.append(nn.Conv1d(
                in_channels = current_ch,
                out_channels = next_ch,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            conv_layers.append(nn.ReLU(inplace=False))
            #ResidualStack For FE
            conv_layers.append(
                ResidualStack(
                    in_channels=next_ch,
                    num_residual_layers=num_residual_layers,
                    num_residual_hiddens=num_residual_hiddens
                )
            )
            conv_layers.append(nn.ReLU(inplace=False))
            
            current_ch = next_ch
            next_ch = current_ch*4
            
            current_length = next_length
            next_length = current_length//2
        
        conv_layers.append(nn.Conv1d(
            in_channels = current_ch,
            out_channels = num_hiddens,
            kernel_size=3, stride=1, padding=1))
        # conv_layers.append(nn.ReLU(inplace=False))
        
        self.num_conv_layers = len(conv_layers)
        self.conv_layers = conv_layers
        
        self.residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )
        self.z_length = current_length
        
    def forward(self, x):
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
        x = self.residual_stack(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self,
                 in_length,
                 z_length,
                 in_channels,
                 out_channels,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels= in_channels,
            out_channels=num_hiddens,
            kernel_size=3, stride=1, padding=1)

        self.residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )
        
        conv_trans_layers = nn.ModuleList()
        current_ch = num_hiddens
        next_ch = None
        current_length = z_length
        next_length = None
        
        while True:
            if next_ch is None:
                next_ch = current_ch//4
                next_length = current_length*2
            if next_ch < 1:
                break
            
            # TransConv For Scale-Up
            conv_trans_layers.append(nn.ConvTranspose1d(
                in_channels = current_ch,
                out_channels = next_ch,
                kernel_size=4,
                stride=2,
                padding=1
            ))            
            conv_trans_layers.append(nn.ReLU(inplace=False))
            conv_trans_layers.append(
                ResidualStack(
                    in_channels=next_ch,
                    num_residual_layers=num_residual_layers,
                    num_residual_hiddens=num_residual_hiddens
                )
            )
            conv_trans_layers.append(nn.ReLU(inplace=False))
            
            current_ch = next_ch
            next_ch = current_ch//4
        
        conv_trans_layers.append(nn.ConvTranspose1d(
            in_channels = current_ch,
            out_channels = out_channels,
            kernel_size=3, stride=1, padding=1))
        # conv_trans_layers.append(nn.ReLU(inplace=False))
        
        self.num_conv_layers = len(conv_trans_layers)
        self.conv_trans_layers = conv_trans_layers
        
    def forward(self, x):
        x = self.conv(x)
        
        x = self.residual_stack(x)
        
        for i in range(self.num_conv_layers):
            x = self.conv_trans_layers[i](x)
        
        return x

class ChannelWiseFC(nn.Module):
    def __init__(self, in_channels, in_features, out_features):
        super(ChannelWiseFC, self).__init__()
        
        # 채널마다 고유의 FC Layer를 생성
        self.fcs = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(in_channels)
        ])
        
    def forward(self, x):
        # 입력 형태: (batch_size, in_channels, in_features)
        
        # 각 채널마다 FC 레이어를 적용하기 위해 채널마다 따로 계산
        out = []
        for i, fc in enumerate(self.fcs):
            # x[:, i, :]의 크기: (batch_size, in_features)
            # fc(x[:, i, :])의 크기: (batch_size, out_features)
            out.append(fc(x[:, i, :]))
        
        # (batch_size, in_channels, out_features) 형태로 결과를 합침
        out = torch.stack(out, dim=1)
        
        return out