import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, channel, kernel_size=3):
        """
        Residual Block( under 50layers + full pre-activation version )
        - under 50 은 middle dim 없이 2개의 동일한 컨볼루션을 사용한다
        - full pre-activation 은 연산전에 batch norm가 relu를 먼저한다

        Args:
            channel (int): number of input channel
            kernel_size (int, optional): kernel size for convolution Defaults to 3.
        """
        super(ResidualBlock1D, self).__init__()
        
        stride = 1
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(channel)
        self.bn2 = nn.BatchNorm1d(channel)
        
    def forward(self, x):

        residual = self.bn1(x)
        residual = self.relu(residual)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        
        out = x + residual
        
        return out
    
class ResidualLayer1D(nn.Module):
    def __init__(self, in_channel, out_channel, num_layer, layer_name, kernel_size=3, is_down_conv=False, is_up_conv=False):
        super(ResidualLayer1D, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            block = ResidualBlock1D(
                channel=in_channel,
                kernel_size=kernel_size
            )
            block_name =f'{layer_name}_{i+1}'
            setattr(self, block_name, block)
            self.layers.append(block)
        

        if is_down_conv:
            self.last_conv = nn.Conv1d(in_channels=in_channel, 
                                       out_channels=out_channel, 
                                       kernel_size=3, 
                                       stride=2,
                                       padding=1)
        elif is_up_conv:
            self.last_conv = nn.ConvTranspose1d(in_channels=in_channel, 
                                       out_channels=out_channel, 
                                       kernel_size=3, 
                                       stride=2,
                                       padding=1,
                                       output_padding=1)
        else:
            self.last_conv = nn.Conv1d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.last_conv(x)
        return x
    
class Encoder1D(nn.Module):
    def __init__(self, in_channel):
        super(Encoder1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channel,
                                out_channels=64,
                                kernel_size=7,
                                stride=2,
                                padding=2)

        self.conv2 = ResidualLayer1D(
            in_channel=64,
            out_channel=128,
            num_layer=3,
            layer_name='conv2',
            is_down_conv=True
        )
        self.conv3 = ResidualLayer1D(
            in_channel=128,
            out_channel=256,
            num_layer=3,
            layer_name='conv3',
            is_down_conv=True
        )
        self.conv4 = ResidualLayer1D(
            in_channel=256,
            out_channel=512,
            num_layer=3,
            layer_name='conv4',
            is_down_conv=True
        )
        self.conv5 = ResidualLayer1D(
            in_channel=512,
            out_channel=512,
            num_layer=3,
            layer_name='conv5',
            is_down_conv=False
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
    
class Decoder1D(nn.Module):
    def __init__(self, in_channel):
        super(Decoder1D, self).__init__()
        
        

        self.conv1 = ResidualLayer1D(
            in_channel=512,
            out_channel=512,
            num_layer=3,
            layer_name='conv2',
            is_up_conv=True
        )
        self.conv2 = ResidualLayer1D(
            in_channel=512,
            out_channel=256,
            num_layer=4,
            layer_name='conv3',
            is_up_conv=True
        )
        self.conv3 = ResidualLayer1D(
            in_channel=256,
            out_channel=128,
            num_layer=6,
            layer_name='conv4',
            is_up_conv=True
        )
        self.conv4 = ResidualLayer1D(
            in_channel=128,
            out_channel=64,
            num_layer=3,
            layer_name='conv5',
            is_up_conv=False
        )
        self.conv5 = nn.ConvTranspose1d(in_channels=64,
                                out_channels=in_channel,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                output_padding=1)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x

class AutoEncoder1D(nn.Module):
    def __init__(self, input_len=1024, input_channel=2, z_dim=512, style_layers=['conv5'], content_layers=['conv1', 'conv2', 'conv3', 'conv4', 'conv5'], 
                    content_weight=0.5, style_weight=0.5):
        super(AutoEncoder1D, self).__init__()


        self.enc = Encoder1D(in_channel=input_channel)
        self.dec = Decoder1D(in_channel=input_channel)

        middle_dim = 512
        middle_len = input_len//16


        self.bottle_enc = nn.ModuleList()
        bottle_dec = []
        current_dim = middle_dim
        current_len = middle_len
        while True:
            
            next_len = current_len//2
            if next_len==0:
                next_len=1
                
            next_dim = current_dim*2
            if next_dim > z_dim:
                next_dim=z_dim
            
            conv = nn.Conv1d(
                        in_channels=current_dim,
                        out_channels=next_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1
                        )
            deconv = nn.ConvTranspose1d(
                        in_channels=next_dim,
                        out_channels=current_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                        )
            self.bottle_enc.append(conv)
            self.bottle_enc.append(nn.ReLU(inplace=True))
            self.bottle_enc.append(nn.BatchNorm1d(num_features=next_dim))
            bottle_dec.append(deconv)
            bottle_dec.append(nn.ReLU(inplace=True))
            bottle_dec.append(nn.BatchNorm1d(num_features=next_dim))
            
            if next_dim==z_dim and next_len==1:
                self.bottle_enc.append(nn.Flatten())
                bottle_dec.append(nn.Unflatten(dim=1, unflattened_size=(z_dim,1)))
                bottle_dec.reverse()
                self.bottle_dec =nn.ModuleList(bottle_dec)
                break

            current_dim = next_dim
            current_len = next_len

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        


    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)

        return recon
    
    def encode_middle_features(self, x, target_layers, layer_names=['conv1', 'conv2', 'conv3','conv4','conv5']):
        middle_feature = x
        
        relu = nn.ReLU(inplace=True)
        
        out_dict = {}
        for layer_name in layer_names:
            layer = getattr(self.enc, layer_name)
            middle_feature = layer(middle_feature)
            if layer_name in target_layers:
                out_dict[layer_name] = relu(middle_feature)
        
        return out_dict
    
    def encode(self, x, before_vector=True):
        
        middle_feature = self.enc(x)
        
        if before_vector:
            return middle_feature

        else:
            for layers in self.bottle_enc:
                middle_feature = layers(middle_feature)

            return middle_feature
    def decode(self, z, before_vector=True):
        middle_feature = z
        if before_vector:
            recon = self.dec(middle_feature)
            
            return recon
        else:
            for layers in self.bottle_dec:

                middle_feature = layers(middle_feature)
            recon = self.dec(middle_feature)
            
            return recon
        
    
    def recon_loss(self, x, with_recon=True):
        
        mse_loss = nn.MSELoss()
        recon = self.forward(x)
        recon_loss = mse_loss(recon, x)
        recon_loss = recon_loss.sum()

        if with_recon:
            return recon_loss, recon
        else:
            return recon_loss
    
    def style_content_loss(self, style_input, content_input, transfer_out):

        # 1. Calculate Content Loss
        total_content_loss = 0.0
        contnet_features = self.encode_middle_features(content_input, target_layers=self.content_layers)
        transfer_features = self.encode_middle_features(transfer_out,target_layers=self.content_layers)

        content_layer_loss = [nn.MSELoss()]*len(self.content_layers)
        for idx, layer_name in enumerate(self.content_layers):
            content_middle_feature = contnet_features[layer_name]
            content_middle_feature = content_middle_feature.detach()
            transfer_middle_feature = transfer_features[layer_name]
            
            loss = content_layer_loss[idx](transfer_middle_feature, content_middle_feature)
            total_content_loss += loss
        total_content_loss = total_content_loss* self.content_weight

        # 2. Calculate Style Loss
        total_style_loss = 0.0
        style_features = self.encode_middle_features(style_input, target_layers=self.style_layers)
        transfer_features = self.encode_middle_features(transfer_out, target_layers=self.style_layers)

        style_layer_loss = [nn.MSELoss()]*len(self.style_layers)
        for idx, layer_name in enumerate(self.style_layers):
            style_middle_feature = style_features[layer_name]
            style_middle_feature = style_middle_feature.detach()
            transfer_middle_feature = transfer_features[layer_name]
            
            loss = style_layer_loss[idx](transfer_middle_feature, style_middle_feature)
            total_style_loss += loss
        total_style_loss = total_style_loss* self.style_weight

        return total_content_loss, total_style_loss
    
if __name__ =='__main__':
    
    # 1. Encoder 와 Decoder 동작 확인
    sample_input = torch.randn(4,2,1024)
    print(f'sample input : {sample_input.shape}')
    enc = Encoder1D(in_channel=2)
    dec = Decoder1D(in_channel=2)
    sample_feature = enc(sample_input)
    print(f'sample_feature : {sample_feature.shape}')
    sample_recon = dec(sample_feature)
    print(f'sample_recon : {sample_recon.shape}')
    
    # 2. AutoEncoder 의 동작 확인
    input_len=1024
    input_channel=2
    z_dim = 512

    sample_input = torch.randn(4, input_channel, input_len)
    model = AutoEncoder1D(input_len=input_len, input_channel=input_channel, z_dim=z_dim)
    recon = model(sample_input)
    
