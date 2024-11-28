import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16



class FeedForward(nn.Module): 
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), 
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64): 
        super().__init__()
        inner_dim = dim_head * heads 
        self.heads = heads
        self.scale = dim ** -0.5 

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):  
        b, n, _, h = *x.shape, self.heads  
        qkv = self.to_qkv(x).chunk(3, dim = -1)  
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) 

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale 



        attn = dots.softmax(dim=-1) 

        out = torch.einsum('bhij,bhjd->bhid', attn, v) 
        out = rearrange(out, 'b h n d -> b n (h d)') 
        out = self.to_out(out) 
        return out

class CAAttention(nn.Module):
    def __init__(self, channels, heads =4):
        super(CAAttention, self).__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))  # h11

        self.query_filter =  nn.Conv2d(channels, channels, kernel_size=(1, 1))
            
        self.key_filter =  nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.value_filter =  nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.project_out = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, x): 
        x1 = x.transpose(1,2).unsqueeze(-1) 
        B, C, N, _ = x1.shape
        q = self.query_filter(x1) 
        k = self.key_filter(x1) 
        v = self.value_filter(x1)  


        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads) 
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads) 
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads) 

        q = torch.nn.functional.normalize(q, dim=-1) 
        k = torch.nn.functional.normalize(k, dim=-1) 

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1) 
        out = (attn @ v)  

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=N, w=1) #bcn1

        out = self.project_out(out) 
        out = out.squeeze(-1) #bcn
        out = out.transpose(1,2)
        return out + x



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        nn.Module.__init__(self)
        self.fc_ca1 = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim, heads = heads, dim_head = dim_head)
        )
        self.fc_sa1 = nn.Sequential(
            nn.LayerNorm(dim),
            CAAttention(dim, heads = heads)
        )
        
        self.fc_ca2 = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim, heads = heads, dim_head = dim_head)
        )
        self.fc_sa2 = nn.Sequential(
            nn.LayerNorm(dim),
            CAAttention(dim, heads = heads)
        )
        self.ff0 = FeedForward(dim, mlp_dim)
        self.ff1 = FeedForward(dim, mlp_dim)

    def forward(self, x): #x=b*(n+1)*c 
        x_ca1 = self.fc_ca1(x)
        x_ca1 = self.ff0(x_ca1)
        x_ca1 = x_ca1 + x
        x_sa1 = self.fc_sa1(x_ca1)
        x_sa1 = self.ff1(x_sa1)
        x_sa1 = x_ca1 + x_sa1
        
        x_ca2 = self.fc_ca2(x_sa1)
        x_ca2 = self.ff0(x_ca2)
        x_ca2 = x_ca2 + x_sa1
        x_sa2 = self.fc_sa2(x_ca2)
        x_sa2 = self.ff1(x_sa2)
        x_sa2 = x_ca2 + x_sa2

        return x_sa2

