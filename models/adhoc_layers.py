import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ChannelWiseFC

class EqualizedLearningRate(nn.Module):
    """ Equalized Learning Rate layer for StyleGAN. """
    def __init__(self, layer):
        super(EqualizedLearningRate, self).__init__()
        self.layer = layer
        nn.init.normal_(self.layer.weight, 0.0, 1.0)  # 가중치를 정규 분포로 초기화
        self.scale = (torch.mean(self.layer.weight.data ** 2).sqrt() + 1e-8)  # 평균 제곱근으로 스케일 계산
        self.layer.weight.data /= self.scale  # 가중치에 스케일 적용
    
    def forward(self, x):
        return self.layer(x) * self.scale  # forward에서 스케일을 적용하여 학습

class MappingNetwork(nn.Module):
    def __init__(self, 
                num_embeddings,
                embedding_dim, 
                num_layers=8):
        super(MappingNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        z_dim = int(num_embeddings*embedding_dim)
        
        self.num_layers = num_layers
        
        # FC 레이어를 구성
        self.fc_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(z_dim, z_dim)
            nn.init.kaiming_normal_(layer.weight, a=0.2, nonlinearity='leaky_relu')  # Leaky ReLU 활성화 함수에 맞춘 초기화
            layer = EqualizedLearningRate(layer)
            self.fc_layers.append(layer)
            
        self.inst_norms = nn.ModuleList([nn.InstanceNorm1d(affine=False) for _ in range(num_layers)])

    def forward(self, z):
        
        batch_size = z.shape[0]
        z = z.view(batch_size, -1)
        w = z
        
        for i in range(self.num_layers):
            w = self.fc_layers[i](w)
            w = self.inst_norms[i](w.unsqueeze(1)).squeeze(1)  # Instance Norm을 사용하여 정규화
            w = F.leaky_relu(w, 0.2)  # Leaky ReLU 활성화 함수
        
        w = w.view(batch_size, self.num_embeddings, self.embedding_dim)    
        
        return w

class Classifier(nn.Module):
    def __init__(self, in_channels, in_dim, n_classes):
        super(Classifier, self).__init__()
        
        self.squeeze_layers = nn.Sequential(ChannelWiseFC(in_channels=in_channels,
                                    in_features=in_dim,
                                    out_features=int(in_dim//2)),
                                nn.ReLU(),
                                nn.BatchNorm1d(in_dim),
                                ChannelWiseFC(in_channels=in_channels,
                                    in_features=int(in_dim//2),
                                    out_features=1),
                                nn.Flatten()
                                )
        self.classifier = nn.Sequential(
                                nn.Linear(in_channels, in_channels//2),
                                nn.ReLU(),
                                nn.Linear(in_channels//2, n_classes)
                                )
        
        nn.Flatten()        
    def forward(self, x):
        
        z = self.squeeze_layers(x)
        out = self.classifier(z)
        
        return out

if __name__ == '__main__':
    print('StyleGAN Idea')