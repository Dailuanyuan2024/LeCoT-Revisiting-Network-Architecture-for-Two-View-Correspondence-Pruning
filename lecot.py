import torch
import torch.nn as nn
from loss import batch_episym
from Transformer import Transformer
from einops import rearrange, repeat


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)

class TransM(nn.Module):
    def __init__(self, in_channel, out_channel, p_size,  T_depth, heads, dim_head, mlp_dim):
        super(TransM, self).__init__()
        self.p_size = p_size #1
        self.patch_to_embedding = nn.Linear(in_channel, out_channel) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel)) 
        # self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim)

    def forward(self, x):
        _,_,hh,ww = x.size() 
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size) 
        x = self.patch_to_embedding(x) 
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) 
        x = torch.cat((cls_tokens, x), dim=1) 
        # x = self.dropout(x) #b*n+1*c
        x = self.transformer(x)  
        x = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=hh, w=ww) 
        return x 

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
  
    mask = logits[:, 0, :, 0] 
    weights = logits[:, 1, :, 0] 

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class LeCoT_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, sampling_rate=0.5):
        super(LeCoT_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)), #4或6 → 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        
        self.l0 = []
        for _ in range(2):
            self.l0.append(TransM(in_channel=128, out_channel=128, p_size=1,  T_depth=2,
                             heads=4, dim_head=32, mlp_dim=128))
        
        self.l1 = []
        for _ in range(3):
            self.l1.append(TransM(in_channel=128, out_channel=128, p_size=1,  T_depth=2,
                             heads=4, dim_head=32, mlp_dim=128))
        self.l0 = nn.Sequential(*self.l0)
        self.l1 = nn.Sequential(*self.l1)

        self.embed_01 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_01 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_1 = nn.Sequential(
            nn.Conv2d(self.out_channel, 64, kernel_size=1), nn.InstanceNorm2d(64, eps=1e-3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1), nn.InstanceNorm2d(16, eps=1e-3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1), nn.InstanceNorm2d(4, eps=1e-3), nn.BatchNorm2d(4), nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)] 
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices) 
            w_out = torch.gather(weights, dim=-1, index=indices) 
        indices = indices.view(B, 1, -1, 1) 

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) 
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4)) 
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1)) 
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        # x_[32,1,1000,6],y1[32,1000]
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous() 
        out = self.conv(out) #out=bcn1
        
        x0 = self.l0(out) #BC250*1    
        out0 = self.embed_01(x0)
        w0 = self.linear_01(out0).view(B, -1) #w1[32,2000]
 
        x1 = self.l1(x0) #BC250*1
        out1 = self.embed_1(x1)
        w1 = self.linear_1(out1).view(B, -1) #w1[32,2000]

        if self.predict == False: 
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) 
            w1_ds = w1_ds[:, :int(N*self.sr)] 
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else: #剪枝，出预测结果
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True) 
            w1_ds = w1_ds[:, :int(N*self.sr)] 
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            
            out = self.embed_2(out)
            w2 = self.linear_2(out) 
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class LeCoT(nn.Module):
    def __init__(self, config):
        super(LeCoT, self).__init__()

        self.LeCoT_0 = LeCoT_Block(initial=True, predict=False, out_channel=128, sampling_rate=config.sr)#sampling_rate=0.5
        self.LeCoT_1 = LeCoT_Block(initial=False, predict=True, out_channel=128, sampling_rate=config.sr)

    def forward(self, x, y):
        #x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.LeCoT_0(x, y) 

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1) 
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1) 
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1) 

        x2, y2, ws1, w_ds1, e_hat = self.LeCoT_1(x_, y1) 

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat) 
        #print(y_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


