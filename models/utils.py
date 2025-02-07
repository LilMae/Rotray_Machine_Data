import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE,self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2*z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, mu, logvar):
        mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)

        return mu + std

    def forward(self,x):


        z_q_mu, z_q_logvar = self.encode(x).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu, z_q_logvar)
        out = self.decode(z_q)

        KL = torch.sum(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)

        return out, KL
    
class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()

        self.convs = nn.Sequential(nn.Conv1d(512,256,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(256,128,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(128,matrixSize,3,1,1))

        # 32x8x8
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        
        self.matrixSize = matrixSize
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,l = out.size()
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(l)
        out = self.flatten(out)
        out = self.fc(out)
        
        out = out.view(out.size(0), self.matrixSize, self.matrixSize)
        
        return out

class LinearTransformationModule(nn.Module):
    def __init__(self, matrix_size=32, z_dim=512):
        super(LinearTransformationModule, self).__init__()
        self.style_net = CNN(matrixSize=matrix_size)
        self.content_net = CNN(matrixSize=matrix_size)
        self.compress = nn.Conv1d(z_dim, matrix_size, 1,1,0)
        self.decompress = nn.Conv1d(matrix_size, z_dim, 1,1,0)
        self.vae = VAE(z_dim=z_dim)

    def forward(self, content_feature, style_feature, transfer=True):

        # 1. Content mean and Variance
        content_mean = torch.mean(content_feature, dim=2,keepdim=True)
        content_mean = content_mean.expand_as(content_feature)
        content_feature = content_feature - content_mean

        # 2. Style mean and KL-Div
        style_mean = torch.mean(style_feature, dim=2, keepdim=True)
        style_mean = style_mean.squeeze(-1)
        style_mean, KL = self.vae(style_mean)
        style_mean = style_mean.unsqueeze(-1)
        style_mean = style_mean.expand_as(style_feature)
        style_feature = style_feature - style_mean


        # Option 1 : Style Transfer
        if transfer:
            style_matrix = self.style_net(style_feature)
            content_matrix = self.content_net(content_feature)

            transmatrix = torch.bmm(style_matrix, content_matrix)
            compressed_content = self.compress(content_feature)
            transfeature = torch.bmm(transmatrix, compressed_content)

            out = self.decompress(transfeature)
            out = out + style_mean
            
            return out, transmatrix, KL

        # Option 2 : Not Style Transfer
        else:
            compressed_content = self.compress(content_feature)
            out = self.decompress(compressed_content)
            out = out + style_mean
            
            return out
        
if __name__ == '__main__':
    print('loss')