import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.matching_trainer import MappingNetwork
from models.layers import Encoder, Decoder, ChannelWiseFC

class AE_model(nn.Module):
    def __init__(self,
                num_hiddens,
                in_length,
                in_channels,
                num_residual_layers,
                num_residual_hiddens,
                num_embeddings,
                embedding_dim,
                commitment_cost,
                use_mapping_net=False):
        super(AE_model, self).__init__()
        
        self.in_length =in_length
        self.encoder = Encoder(
            in_channels=in_channels, 
            in_length = in_length,
            num_hiddens=num_hiddens, 
            num_residual_layers=num_residual_layers, 
            num_residual_hiddens=num_residual_hiddens
        )
        z_length = self.encoder.z_length
        self.pre_vq_conv = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_embeddings,
            kernel_size=1,
            stride=1
        )
        self.pre_vq_fc = ChannelWiseFC(
            in_channels = num_embeddings,
            in_features = z_length,
            out_features = embedding_dim
        )
        self.use_mapping_net = use_mapping_net
        if self.use_mapping_net:
            self.mapping_net = MappingNetwork(
                num_embeddings,
                embedding_dim, 
                num_layers=8
            )
        
        
        self.after_vq_fc = ChannelWiseFC(
            in_channels = num_embeddings,
            in_features = embedding_dim,
            out_features = z_length
        )
        
        
        self.decoder = Decoder(
            in_channels=num_embeddings, 
            in_length=in_length,
            out_channels=in_channels,
            z_length=z_length,
            num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens
        )
        
    def forward(self, x):
        """
        z (batch x num_embeddings x embedding_dim)
        """
        
        z = self.encoder(x)
        z = self.pre_vq_conv(z)     
        z = self.pre_vq_fc(z)
        if self.use_mapping_net:
            z = self.mapping_net(z)
        z = self.after_vq_fc(z)        
        x_recon = self.decoder(z)

        return x_recon
    
    def tesnor_transform_checker(self, x):
        """
        z (batch x num_embeddings x embedding_dim)
        """
        print(f'input shape : {x.shape}')
        z = self.encoder(x)
        print(f'after encoder : {z.shape}')
        z = self.pre_vq_conv(z)     
        print(f'after pre_vq_conv : {z.shape}')
        z = self.pre_vq_fc(z)
        print(f'after pre_vq_fc : {z.shape}')
        if self.use_mapping_net:
            z = self.mapping_net(z)
        z = self.after_vq_fc(z)      
        print(f'after after_vq_fc : {z.shape}')
        x_recon = self.decoder(z)
        print(f'output : {x_recon.shape}')

        return x_recon
    
    def interpolation(self, x0, x1, ratio, before_vq=True):
        z0 = self.encoder(x0)
        z0 = self.pre_vq_conv(z0)
        z0 = self.pre_vq_fc(z0)
        if self.use_mapping_net:
            z0 = self.mapping_net(z0)
        
        z1 = self.encoder(x1)
        z1 = self.pre_vq_conv(z1)    
        z1 = self.pre_vq_fc(z1)
        if self.use_mapping_net:
            z1 = self.mapping_net(z1)
        
        
        z = z0*ratio + z1*(1-ratio)
        
        
        z = self.after_vq_fc(z)
        x_interpolated = self.decoder(z)
        
        return x_interpolated


if __name__ == '__main__':
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 32
    commitment_cost = 0.25
    learning_rate = 1e-3
    in_channels = 2
    in_length = 512

    ae_model = AE_model(
        num_hiddens=num_hiddens,
        in_length = in_length,
        in_channels=in_channels,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost
    )
    
    torch.autograd.set_detect_anomaly(True)
    
    sample_input = torch.randn([4, in_channels, in_length])

    x_recon = ae_model.tesnor_transform_checker(sample_input)

