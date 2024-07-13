import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet


from einops import rearrange, repeat
from einops.layers.torch import Rearrange



import numpy as np
import torch.nn.functional as F

class GANEncoder(nn.Module): # conditional Encoder
    def __init__(self):
        super(GANEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256)),  # Resize to 256x256
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64 x 128 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 64 x 64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 32 x 32
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512 x 16 x 16
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 1024 x 8 x 8
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),  # 512 x 4 x 4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # 256 x 2 x 2
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # 128 x 1 x 1
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.model(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 64 x 128 x 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 128 x 64 x 64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 256 x 32 x 32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 512 x 16 x 16
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 1024 x 8 x 8
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1)  # 512 x 4 x 4
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)  # 256 x 2 x 2
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)  # 128 x 1 x 1
        # self.conv9 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)  # 64 x 1 x 1
        # self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)  # 64 x 1 x 1
        # self.conv9 = nn.Conv2d(64, 181, kernel_size=1, stride=1, padding=0)  # 181 x 1 x 1
        
        # self.norm1 = nn.InstanceNorm2d(64)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)
        self.norm4 = nn.InstanceNorm2d(512)
        self.norm5 = nn.InstanceNorm2d(1024)
        self.norm6 = nn.InstanceNorm2d(512)
        self.norm7 = nn.InstanceNorm2d(256)
        # self.norm8 = nn.InstanceNorm2d(128)
        
    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # Resize 256x256
        x = F.relu(self.conv1(x))
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.norm3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.norm4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.norm5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.norm6(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.norm7(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.conv8(x), 0.2)
        # x = F.leaky_relu(self.norm9(self.conv9(x)), 0.2)
        return x
    
class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

class ResnetEncoder_feat(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder_feat, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return features,parameters

# from transformers import BertModel, BertConfig

class DetailAUTransformer(nn.Module):
    def __init__(self, detail_dim=2048, au_dim=41, hidden_dim=256, output_dim=128, num_layers=4, num_heads=8):
        super(DetailAUTransformer, self).__init__()
        
        # Embedding layers
        self.detail_embedding = nn.Linear(detail_dim, hidden_dim)
        self.au_embedding = nn.Linear(au_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder for detail parameters
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, detail_features, au_features):
        batch_size = detail_features.size(0)
        
        # Embedding
        detail_embedded = self.detail_embedding(detail_features)
        au_embedded = self.au_embedding(au_features)
        
        # Combine and add positional embeddings
        x = torch.concat([detail_embedded, au_embedded], dim=1)
        x += self.pos_embedding
        
        # Transformer Encoder
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_dim)
        
        # Use only the [CLS] token (first token) for regression
        x = x[:, 0, :]
        
        # Decode to detail parameters
        detail_params = self.decoder(x)
        
        return detail_params



# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)
        # print(x.shape, y.shape)
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        # print(q.shape)
        # print(kv)
        # print(kv.shape)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print(qkv[0].shape,q.shape,k.shape, v.shape ) # torch.Size([1, 4, 256]) torch.Size([1, 4, 4, 64]) torch.Size([1, 4, 4, 64]) torch.Size([1, 4, 4, 64])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, y):
        for attn, cattn, ff in self.layers:
            x = attn(x) + x
            x = cattn(x, y) + x
            x = ff(x) + x

        return self.norm(x)




class TDC(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, alpha, nheads, batchsize, detail_dim=181, au_dim=41):
        super(TDC, self).__init__()
        self.dropout = dropout
        self.batchsize = batchsize
        self.detail_embedding = nn.Linear(detail_dim, nhid)
        self.au_embedding = nn.Linear(au_dim, nhid)
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, batchsize=self.batchsize) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 42, nhid))
        self.cls_token = nn.Parameter(torch.randn(1, 1, nhid))
        self.Tdropout = nn.Dropout(0.0)

        self.transformer = Transformer(nhid, 6, nheads, 64, 256, 0.0)
        self.pool = 'cls'
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(nhid, noutput)
        nn.init.xavier_uniform_(self.mlp_head.weight, gain=0.01)
        
    def forward(self, x, y):
       
        b, n = x.shape

        # x = x.unsqueeze(1)

        x = self.au_embedding(x)
        y = self.detail_embedding(y)
        
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.Tdropout(x)
        
        x = self.transformer(x, y)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
class AUEncoder(nn.Module):
    def __init__(self):
        super(AUEncoder,self).__init__()

        self.flatten=nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(27*512, 50)

    def forward(self, x):
        x=self.flatten(x)
        x=self.fc(x)
        return x