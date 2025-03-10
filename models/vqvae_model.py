import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.adhoc_layers import MappingNetwork
from models.layers import Encoder, Decoder, ChannelWiseFC

class VectorQuntizer(nn.Module):
    def __init__(self,
                num_embedding,
                embedding_dim,
                commitment_cost):
        """
        Encoder에서 인코딩한 z_e(x) 를 
        Lookup CodeBook 과정을 통해, z_q(x) 로 바꾸어줌

        + 후에 학습을 위해, codebook loss와 commitment loss 를 연산함

        Args:
            num_embedding (int): code book에서 사용하는 embedding 개수
            embedding_dim (int): 각 embedding의 차원(깊이)
            commitment_cost (float): commitment loss에 곱해지는 가중치(0 ~ 1.0)
        """
        super(VectorQuntizer, self).__init__()
        
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embedding,
                                            1/self.num_embedding)
    
    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        
        #  After Step 1: Encoding
        inputs = inputs.permute(0,2,1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Step 2 : Distance Measuring
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2* torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        # Step 3 : Select Code Index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embedding, device = inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Step 4 : LookPu code book
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost*e_latent_loss
        
        quantized = inputs + (quantized-inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape encoding_indices to match the original input shape
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return loss, quantized.permute(0,2,1).contiguous(), perplexity, encoding_indices
    
class VectorQuntizerEMA(nn.Module):
    def __init__(self,
                num_embedding,
                embedding_dim,
                commitment_cost,
                decay,
                epsilon=1e-5):
        """
        Online Learning 방식에서는 CodeBook Loss 를 통해 바로 최적값을 구할 수 없다.
        따라서 EMA(Exponential Moving Average) 방법을 적용한 CodeBook Loss 를 구한다. 
        
        EMA : Moving Average 인데, 멀리 있는 값일 수록 낮은 가중치를 주어 가중 평균을 구한다.

        Args:
            num_embedding (int): code book에서 사용하는 embedding 개수
            embedding_dim (int): 각 embedding의 차원(깊이)
            commitment_cost (float): commitment loss에 곱해지는 가중치(0 ~ 1.0)
            
            decay (float): 거리에 따른 가중치 감쇠
            epsilon (float, optional): Defaults to 1e-5.
        """

        super(VectorQuntizerEMA, self).__init__()
        
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embedding,
                                            1/self.num_embedding)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embedding))
        self.ema_w = nn.Parameter(torch.Tensor(num_embedding, self.embedding_dim))
        self.ema_w.data.normal_()
        
        self.decay = decay
        self.epsilon = epsilon
        
    
    def forward(self, inputs):
        # convert inputs from BCL -> BLC

        #  After Step 1: Encoding

        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Step 2 : Distance Measuring
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2* torch.matmul(flat_input, self.embedding.weight.t())
        )
        
        # Step 3 : Select Code Index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embedding, device = inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Step 4 : LookPu code book
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # EMA  : Epsilon Moving Average
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                (1-self.decay) * torch.sum(encodings, 0)

            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_embedding * self.epsilon) * n)
        
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w*self.decay + (1 - self.decay)*dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized-inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape encoding_indices to match the original input shape
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        
        return loss, quantized, perplexity, encoding_indices
    
class VQVAE_model(nn.Module):
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
        super(VQVAE_model, self).__init__()
        
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
        
        
        self.vq_vae = VectorQuntizerEMA(num_embedding=num_embeddings,
                                        embedding_dim=embedding_dim,
                                        commitment_cost=commitment_cost, decay=0.99)
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
    
        vq_loss, quantized, perplexity, _ = self.vq_vae(z)

        quantized = self.after_vq_fc(quantized)
        
        x_recon = self.decoder(quantized)

        return x_recon, vq_loss
    
    def tesnor_transform_checker(self, x):
        """
        z (batch x num_embeddings x embedding_dim)
        """
        print(f'input : {x.shape}')
        z = self.encoder(x)
        print(f'after encoder : {z.shape}')
        z = self.pre_vq_conv(z)   
        print(f'after pre_vq_conv : {z.shape}')
        z = self.pre_vq_fc(z)
        print(f'after pre_vq_fc : {z.shape}')
        if self.use_mapping_net:
            z = self.mapping_net(z)
    
        vq_loss, quantized, perplexity, _ = self.vq_vae(z)
        print(f'after vq_vae : {quantized.shape}')
        quantized = self.after_vq_fc(quantized)
        print(f'after after_vq_fc : {quantized.shape}')
        x_recon = self.decoder(quantized)
        print(f'output : {x_recon.shape}')
        return x_recon, vq_loss
    
    def encode(self, x, for_z=True):
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
    
    vqvae_model = VQVAE_model(
        num_hiddens=num_hiddens,
        in_length = in_length,
        in_channels = in_channels,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost
    )
    
    torch.autograd.set_detect_anomaly(True)
    
    sample_input = torch.randn([4, in_channels, in_length])

    x_recon, vq_loss = vqvae_model.tesnor_transform_checker(sample_input)

    vq_loss.backward()
    print('Backward Success')