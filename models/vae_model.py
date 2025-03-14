import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.adhoc_layers import MappingNetwork
from models.layers import Encoder, Decoder, ChannelWiseFC
    


class VAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, z_dim):
        super(VAE,self).__init__()

        # B  x num_embedding x embedding_dim -> B x num_embedding x embedding_dim//2 -> B x z_dim
        middle_len = num_embeddings * embedding_dim//2
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.encode = nn.Sequential(
            nn.Conv1d(
                in_channels=num_embeddings,
                out_channels=num_embeddings,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm1d(num_embeddings),
            nn.Flatten(),
            nn.Linear(middle_len, 2*z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(
            nn.Linear(z_dim, middle_len),
            nn.Unflatten(dim=1, unflattened_size=(num_embeddings, embedding_dim//2)),
            nn.ConvTranspose1d(
                in_channels=num_embeddings,
                out_channels=num_embeddings,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.BatchNorm1d(num_embeddings)
        )


    def reparameterize(self, mu, logvar):
        mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self,x):

        z_q_mu, z_q_logvar = self.encode(x).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu, z_q_logvar)
        
        out = self.decode(z_q)

        KL = torch.sum(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)

        return out, KL

class VAE_model(nn.Module):
    def __init__(self,
                num_hiddens,
                in_length,
                in_channels,
                z_dim,
                num_residual_layers,
                num_residual_hiddens,
                num_embeddings,
                embedding_dim,
                commitment_cost,
                use_mapping_net=False):
        super(VAE_model, self).__init__()
        
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
        
        
        self.vae = VAE(num_embeddings=num_embeddings,
                        embedding_dim=embedding_dim,
                        z_dim=z_dim)
        
        
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
    
        z, KL = self.vae(z)

        z = self.after_vq_fc(z)
        
        x_recon = self.decoder(z)

        return x_recon, KL
    
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
        z, KL = self.vae(z)
        print(f'after vae : {z.shape}')
        z = self.after_vq_fc(z)      
        print(f'after after_vq_fc : {z.shape}')
        x_recon = self.decoder(z)
        print(f'output : {x_recon.shape}')

        return x_recon, KL
    
    def encode(self,x, for_z=True):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        z = self.pre_vq_fc(z)
        if self.use_mapping_net:
            z = self.mapping_net(z)
        
        return z

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
    
    z_dim = 512

    vae_model = VAE_model(
        num_hiddens=num_hiddens,
        in_length = in_length,
        in_channels=in_channels,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        num_embeddings=num_embeddings,
        z_dim=z_dim,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost
    )
    
    
    torch.autograd.set_detect_anomaly(True)
    
    sample_input = torch.randn([4, in_channels, in_length])

    x_recon, KL = vae_model.tesnor_transform_checker(sample_input)

    KL.backward()
    print('Backward Success')