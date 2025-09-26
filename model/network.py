from RA_layer import RA_Layer1,RA_Layer
import  torch.nn as nn
import torch
from einops import repeat
from torch import einsum
from torch_edge import DilatedKnnGraph
from torch_vertex import GraphConv2d,DenseDynBlock2d,DynConv2d
from torch_nn import BasicConv
import torch.nn.functional as F
from util import  farthest_point_sample ,index_points
from util import grouping
from torch_nn import batched_index_select1
class mlp(nn.Module):
    def __init__(self, in_channels, layer_dim):
        super(mlp, self).__init__()
        self.mlp_list = nn.ModuleList()
        for i, num_outputs in enumerate(layer_dim[:-1]):
            if i == 0:
                sub_module = nn.Sequential(
                    nn.Linear(in_channels, num_outputs),
                    nn.BatchNorm1d(num_outputs),
                    nn.Dropout(p=0.5),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
            else:
                sub_module = nn.Sequential(
                    nn.Linear(layer_dim[i - 1], num_outputs),
                    nn.BatchNorm1d(num_outputs),
                    nn.Dropout(p=0.5),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
        self.mlp_list.append(
            nn.Linear(layer_dim[-2], layer_dim[-1])
        )

    def forward(self, inputs):
        net = inputs
        for sub_module in self.mlp_list:
            net = sub_module(net)
        return net
class mlp_conv(nn.Module):
    def __init__(self, in_channels, layer_dim):
        super(mlp_conv, self).__init__()
        self.conv_list = nn.ModuleList()
        for i, num_out_channel in enumerate(layer_dim[:-1]):
            if i == 0:
                sub_module = nn.Sequential(
                   nn.Conv1d(in_channels=in_channels, out_channels=num_out_channel, kernel_size=1),
                   nn.BatchNorm1d(num_out_channel),
                   nn.ReLU()
                )
                self.conv_list.append(sub_module)
            else:
                sub_module = nn.Sequential(
                   nn.Conv1d(in_channels=layer_dim[i - 1], out_channels=num_out_channel, kernel_size=1),
                   nn.BatchNorm1d(num_out_channel),
                   nn.ReLU()
                )
                self.conv_list.append(sub_module)
        self.conv_list.append(
           nn.Conv1d(in_channels=layer_dim[-2], out_channels=layer_dim[-1], kernel_size=1)
        )

    def forward(self, inputs):
        net = inputs
        for module in self.conv_list:
            net = module(net)
        return net
def batched_index_select(values ,indices ,dim =1):
    values_dim =values.shape[(dim+1):]
    ## 使用lambda表达式，一步实现。
    # 冒号左边是原函数参数；
    # 冒号右边是原函数返回值；
    #map函数第一部分是一个函数操作，第二部分是一个可迭代的对象，可以是元组，列表，字典等
    values_shape ,indices_shape =map(lambda t:list(t.shape),(values,indices))
    indices =indices[(...,*((None,)*len(values_dim)))]
    indices =indices.expand(*((-1,) *len(indices_shape)) ,*values_dim)
    values_expand_len =len(indices_shape) -(dim +1)
    values =values[(*((slice(None),)*dim),*((None,) *values_expand_len),...)]

    values_expand_shape =[-1] *len(values.shape)
    expand_slice =slice(dim ,(dim +values_expand_len))
    values_expand_shape[expand_slice] =indices_shape[expand_slice]
    values =values.expand(*values_expand_shape)

    dim += values_expand_len

    return values.gather(dim ,indices)


def pairwise_distance(x):
    """
       Compute pairwise distance of a point cloud.
       Args:
           x: tensor (batch_size, num_points, num_dims)
       Returns:
           pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner =-2*torch.matmul(x.transpose(2,1),x)
    x_square= torch.sum(x**2 ,dim=1 ,keepdim=True)
    return -x_square-x_inner-x_square.transpose(2,1)
def knn(x,k):
    idx =pairwise_distance(x) #(batchsize ,numpoint ,numpoint)
    idx =idx.topk(k=k,dim=-1)[1] #(batchsize ,numpoint ,k)idx.topk:快速进行排序
    return idx


def get_graph_feature2(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
def get_graph_feature(x ,k=20,idx=None):
    """construct edge feature for each point
        Args:
            tensor: input a point cloud tensor,batch_size,num_dims,num_points   3图卷积
            k: int
        Returns:
            edge features: (batch_size,num_dims,num_points,k)
     """
    batch_size =x.size(0)
    num_point  =x.size(2)
    x =x.view(batch_size ,-1, num_point)
    if idx is None:
        idx =knn(x,k=k)
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =torch.device('cuda')
    idx_base =torch.arange(0,batch_size,device=device).view(-1, 1, 1)*num_point
    idx =idx +idx_base
    idx =idx.view(-1)

    _,num_dim,_ =x.size()
    x =x.transpose(2,1).contiguous()
    feature =x.view(batch_size*num_point, -1)[idx,:]
    feature = feature.view(batch_size, num_point, k, num_dim)
    x = x.view(batch_size, num_point, 1, num_dim).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
def get_graph_feature_local(x ,k=20,idx=None):
    """construct edge feature for each point
        Args:
            tensor: input a point cloud tensor,batch_size,num_dims,num_points
            k: int
        Returns:
            edge features: (batch_size,num_dims,num_points,k)
     """
    batch_size =x.size(0)
    num_point  =x.size(2)
    x =x.view(batch_size ,-1, num_point)
    if idx is None:
        idx =knn(x,k=k)
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =torch.device('cuda')
    idx_base =torch.arange(0,batch_size,device=device).view(-1, 1, 1)*num_point
    idx =idx +idx_base
    idx =idx.view(-1)

    _,num_dim,_ =x.size()
    x =x.transpose(2,1).contiguous()
    feature =x.view(batch_size*num_point, -1)[idx,:]
    feature = feature.view(batch_size, num_point, k, num_dim)
    x = x.view(batch_size, num_point, 1, num_dim).repeat(1, 1, k, 1)
    feature_x =feature - x
    feature = feature_x.permute(0, 3, 1, 2).contiguous()

    return feature

def knn_feature(x ,k=20,idx=None):
    """
     基于knn对特征进行分类
    :param x:
    :param k:
    :param idx:
    :return:
    """
    batch_size = x.size(0)
    num_point = x.size(2)
    x = x.view(batch_size, -1, num_point)
    if idx is None:
      idx = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_point
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dim, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_point, -1)[idx, :]
    feature = feature.view(batch_size,num_dim, num_point,k)
    return feature
def exist(val):
    return val is not None
def max_value(x):
    return torch.finfo(x.dtype).max

class attention_unit(nn.Module):
    def __init__(self,in_channels=130):
        super(attention_unit, self).__init__()
        self.convF =nn.Sequential(
            nn.Conv1d(in_channels=in_channels ,out_channels=in_channels//4,kernel_size=1),
            nn.BatchNorm1d(in_channels//4),
            nn.ReLU()
        )
        self.convG =nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=in_channels//4 ,kernel_size=1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU()
        )
        self.convH =nn.Sequential(
            nn.Conv1d(in_channels=in_channels ,out_channels=in_channels,kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        #self.gamma =nn.Parameter(torch.tensor(torch.zeros([1]))).cpu()
        self.gamma = nn.Parameter(torch.zeros([1],device='cpu'))
    def forward(self,input):
        f =self.convF(input)
        g =self.convG(input)
        h =self.convH(input)
        s =torch.matmul(g.permute(0,2,1) ,f)
        beta =F.softmax(s,dim=2) #b n n
        o =torch.matmul(h ,beta) #b 130 ,n
        x =self.gamma*o +input
        return x
class LocalAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask=None):
        x = x.permute(0, 2, 1)  # transpose
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i=n)  #  (x,y,z)->(x,n,y,z)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest=False)

            v = batched_index_select(v, indices, dim=2)
            qk_rel = batched_index_select(qk_rel, indices, dim=2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)
            mask = batched_index_select(mask, indices, dim=2) if exists(mask) else None

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        agg = agg.permute(0, 2, 1)
        return agg

#特征与相对位置进行结合，修正下采样
class positate_att(nn.Module):
    def __init__(self, *, dim=128, pos_mlp_hidden_dim=64, atten_mlp=2):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.bn =nn.BatchNorm1d(384)
        self.r1 =nn.ReLU()
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
          #  nn.Dropout(p=0.5),
            nn.Linear(pos_mlp_hidden_dim, dim)

        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * atten_mlp),
            nn.ReLU(),
         #   nn.Dropout(p=0.5),
            nn.Linear(dim * atten_mlp, dim)
        )

        self.attn_conv = nn.Sequential(nn.Conv1d(in_channels=dim, out_channels=dim * atten_mlp, kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=dim * atten_mlp, out_channels=dim, kernel_size=1))
    def forward(self,x,pos):
        x = x.permute(0, 2, 1)  # transpose
        n = x.shape[1]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)
        qk_rel = q[:, :, None, :] - k[:, None, :, :]
        v = repeat(v, ' b j d -> b i j d', i=n)  # (x.y,z) ->(x ,n,y,z)
        v = v + rel_pos_emb
        sim = self.attn_mlp(qk_rel + rel_pos_emb)
        # attention
        attn = sim.softmax(dim=-2)
        # aggregate
       # agg = einsum('b i j d ,b i j d ->b i d', attn, v)  # attn *v
        agg =attn*v
        agg = agg.sum(dim=2)
        agg = agg.permute(0, 2, 1)
        return agg
class up_projection_unit(nn.Module):
    def __init__(self, up_ratio=4):
        super(up_projection_unit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=416, out_channels=128, kernel_size=1),

            nn.ReLU(),
            # nn.Conv1d(in_channels=288, out_channels=128, kernel_size=1),#288-352
            # nn.BatchNorm1d(128),
            # nn.ReLU()
        )
        self.up_block1 = up_block11(up_ratio=4, in_channels=128 + 2)
        self.up_block2 = up_block11(up_ratio=4, in_channels=128 + 2)
        self.down_block = down_block11(up_ratio=4, in_channels=128,k=16) #k有16-8

    def forward(self, feature,input,rel_pos_emb):
        L = self.conv1(feature)  # b,128,n
        H0 = self.up_block1(L)  # b,128,n*4
        L0 = self.down_block(H0,input,rel_pos_emb)  # b,128,n
       # L0 = self.down_block(H0)
        E0 = L0 - L  # b,128,n
        H1 = self.up_block2(E0)  # b,128,4*n
        H2 = H0 + H1  # b,128,4*n
        return H2

class up_block(nn.Module):
    def __init__(self,up_ratio=4, in_channels=130):
        super(up_block, self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1),
                nn.ReLU()
            )
        self.conv2 = nn.Sequential(
               nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU()
            )
        self.atten =attention_unit(
              in_channels=in_channels
            )
        self.edge_conv =nn.Sequential(
             nn.Conv2d(128 *2 ,128*self.up_ratio ,kernel_size=1,bias=False),
             nn.BatchNorm2d(512),
             nn.LeakyReLU(negative_slope=0.2)
              )

        self.grid = self.gen_grid(up_ratio).clone().detach().requires_grad_(True)
        self.rayler = RA_Layer1(130)
    def gen_grid(self,up_ratio):
        import math
        sqrted=int(math.sqrt(up_ratio))+1
        for i in range(1,sqrted+1).__reversed__():
            if (up_ratio%i)==0:
                num_x=i
                num_y=up_ratio//i
                break
        grid_x=torch.linspace(-0.2,0.2,num_x)
        grid_y=torch.linspace(-0.2,0.2,num_y)

       # x,y=torch.meshgrid([grid_x,grid_y])
        x, y = torch.meshgrid([grid_x, grid_y],indexing ='ij')
        grid=torch.stack([x,y],dim=-1) # 2,2,2
        grid=grid.view([-1,2])#4,2
        return grid
    def forward(self ,input):
        net =input   #b ,128,n
        grid =self.grid.clone()
        grid =grid.unsqueeze(0).repeat(net.shape[0],1,net.shape[2])  #b ,4,2*n
        grid =grid.view([net.shape[0],-1,2]) #b ,4*n ,2

        x =get_graph_feature(net,k=20)
        net =self.edge_conv(x)
        net =net.max(dim =-1 ,keepdim=True)[0].permute(0,2,1,3)
        net =torch.reshape(net ,(net.shape[0],net.shape[1] *self.up_ratio ,-1))

        net =torch.cat([net ,grid.cuda()],dim=2) #b n*4 ,130
        net =net.permute(0,2,1) #b,130,n*4
       # net =self.atten(net) #b ，130,100  #采用的注意力机制
        net = self.rayler(net)
        net =self.conv1(net)
        net =self.conv2(net)

        return net

class up_block11(nn.Module):
    def __init__(self,up_ratio=4, in_channels=130):
        super(up_block11, self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1),
                nn.ReLU()
            )
        self.conv2 = nn.Sequential(
               nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
               nn.ReLU()
            )
        self.atten =attention_unit(
              in_channels=in_channels
            )
        self.edge_conv =nn.Sequential(
             nn.Conv2d(256*2 ,256 ,kernel_size=1,bias=False),
             nn.BatchNorm2d(512),
             nn.LeakyReLU(negative_slope=0.2)
              )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(256 * 2, 128 * self.up_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.grid = self.gen_grid(up_ratio).clone().detach().requires_grad_(True)
        self.DynConv2d = DynConv2d(in_channels=128, out_channels=256, k=9, conv='edge', act='relu', dilation=1,
                                   norm='batch'
                                   , bias=False, stochastic=False, epsilon=0.2, knn='matrix')

        self.DynConv2d1 = DynConv2d(in_channels=128, out_channels=256, k=9, conv='edge', act='relu', dilation=2,
                                    norm='batch'
                                    , bias=False, stochastic=False, epsilon=0.2, knn='matrix')

        self.DynConv2d2 = DynConv2d(in_channels=256, out_channels=128 * self.up_ratio, k=9, conv='edge', act='relu',
                                    dilation=2, norm='batch'
                                    , bias=False, stochastic=False, epsilon=0.2, knn='matrix')
        self.rayler = RA_Layer1(130)
    def gen_grid(self,up_ratio):
        import math
        sqrted=int(math.sqrt(up_ratio))+1
        for i in range(1,sqrted+1).__reversed__():
            if (up_ratio%i)==0:
                num_x=i
                num_y=up_ratio//i
                break
        grid_x=torch.linspace(-0.2,0.2,num_x)  #0.2-0.1
        grid_y=torch.linspace(-0.2,0.2,num_y)

        x,y=torch.meshgrid([grid_x,grid_y],indexing='ij')
        grid=torch.stack([x,y],dim=-1) # 2,2,2
        grid=grid.view([-1,2])#4,2
        return grid
    def forward(self ,input):
        net =input   #b ,128,n
        grid =self.grid.clone()
        grid =grid.unsqueeze(0).repeat(net.shape[0],1,net.shape[2])  #b ,4,2*n
        grid =grid.view([net.shape[0],-1,2]) #b ,4*n ,2
        x1 = self.DynConv2d(net)
        x2 = self.DynConv2d1(net)
        x = torch.cat([x1, x2], dim=1).unsqueeze(-1)
        # x =torch.cat([x,x1],dim=1)
        x = x.permute(0, 2, 1, 3)
        # x =get_graph_feature(net,k=20)
        # net =self.edge_conv(x)
        # net =net.max(dim =-1 ,keepdim=True)[0].permute(0,2,1,3)
        net = torch.reshape(x, (x.shape[0], x.shape[1] * self.up_ratio, -1))

        net =torch.cat([net ,grid.cuda()],dim=2) #b n*4 ,130
        net =net.permute(0,2,1) #b,130,n*4
       # net =self.atten(net) #b ，130,100  #采用的注意力机制
        net =self.rayler(net)
        net =self.conv1(net)
        net =self.conv2(net)

        return net

class GCN(nn.Module):
    def __init__(self,in_channels=32,out_channels=64,k=20,conv='edge',act='relu',norm='batch',stochastic=True,epsilon=0.2):
        super(GCN, self).__init__()

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels,out_channels, conv, act, norm)
        self.channels = in_channels
    def forward(self, inputs):
        GCN =self.head(inputs, self.knn(inputs[:, 0:self.channels]))

        return GCN


class DenseGCN(nn.Module):
    def __init__(self,channels,c_growth,k=20,conv='edge',act='relu',norm='batch',stochastic=True,epsilon=0.2,knn ='matrix',bias=True):
        super(DenseGCN, self).__init__()

        self.genseGCN =DenseDynBlock2d(channels, c_growth, k, 1, conv, act, norm, bias, stochastic, epsilon, knn)
        self.genseGCN1=DenseDynBlock2d(channels+c_growth, c_growth, k, 1, conv, act, norm, bias, stochastic, epsilon, knn)
        self.genseGCN2 = DenseDynBlock2d(channels + c_growth*2, c_growth, k, 1, conv, act, norm, bias, stochastic,epsilon, knn)
        self.genseGCN3 = DenseDynBlock2d(channels + c_growth * 3, c_growth, k, 1, conv, act, norm, bias, stochastic,
                                         epsilon, knn)

        self.genseGCN4 = DenseDynBlock2d(channels + c_growth * 4, c_growth, k, 1, conv, act, norm, bias, stochastic,
                                         epsilon, knn)

    def forward(self, inputs):
        gensefeature = self.genseGCN(inputs)
        gensefeature_1 = torch.cat([gensefeature, inputs], dim=1)

        gensefeature1 = self.genseGCN1(gensefeature_1)
        gensefeature_2 = torch.cat([gensefeature1, gensefeature_1], dim=1)

        gensefeature2 = self.genseGCN2(gensefeature_2)

       # gensefeature_3 = torch.cat([gensefeature2, gensefeature_2], dim=1)



        # 在增加两个
        # gensefeature3 = self.genseGCN3(gensefeature_3)
        # gensefeature_4 = torch.cat([gensefeature3,inputs], dim=1)
        #
        # gensefeature4 = self.genseGCN4(gensefeature_4)
        # gensefeature_5 = torch.cat([gensefeature4, gensefeature_4], dim=1)

        Reset_feature = torch.cat([gensefeature, gensefeature1, gensefeature2,inputs], dim=1)

        return  Reset_feature
class DenseGCN1(nn.Module):
    def __init__(self,channels,c_growth,k=20,conv='edge',act='relu',norm='batch',dilation=1,stochastic=True,epsilon=0.2,knn ='matrix',bias=True):
        super(DenseGCN1, self).__init__()

        self.genseGCN =DenseDynBlock2d(channels, c_growth, k, dilation, conv, act, norm, bias, stochastic, epsilon, knn)
        self.genseGCN1=DenseDynBlock2d(channels+c_growth, c_growth, k,dilation, conv, act, norm, bias, stochastic, epsilon, knn)
        self.genseGCN2 = DenseDynBlock2d(channels + c_growth*2, c_growth, k,dilation, conv, act, norm, bias, stochastic,epsilon, knn)
        self.genseGCN3 = DenseDynBlock2d(channels + c_growth * 3, c_growth, k, dilation, conv, act, norm, bias, stochastic,
                                         epsilon, knn)

        self.genseGCN4 = DenseDynBlock2d(channels + c_growth * 4, c_growth, k, dilation, conv, act, norm, bias, stochastic,
                                         epsilon, knn)

    def forward(self, inputs):
        gensefeature = self.genseGCN(inputs)
        gensefeature_1 = torch.cat([gensefeature, inputs], dim=1)

        gensefeature1 = self.genseGCN1(gensefeature_1)
        gensefeature_2 = torch.cat([gensefeature1, gensefeature_1], dim=1)

        gensefeature2 = self.genseGCN2(gensefeature_2)

        gensefeature_3 = torch.cat([gensefeature2, gensefeature_2], dim=1)




        # gensefeature3 = self.genseGCN3(gensefeature_3)
        # gensefeature_4 = torch.cat([gensefeature3,inputs], dim=1)
        #
        # gensefeature4 = self.genseGCN4(gensefeature_4)
        # gensefeature_5 = torch.cat([gensefeature4, gensefeature_4], dim=1)

        #Reset_feature = torch.cat([gensefeature, gensefeature1, gensefeature2, inputs], dim=1)

        return  gensefeature_3
class down_block11(nn.Module):
    def __init__(self, up_ratio=4, in_channels=128,k=8):
        super(down_block11, self).__init__()

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.spatial_skip_mlp = nn.Sequential(nn.Conv1d(in_channels+3, 256, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())
        self.up_ratio = up_ratio
        self.k =k
        self.mlp1 =nn.Sequential(nn.Conv1d(in_channels+3,256,kernel_size=1,bias=False),
                                 nn.ReLU())

        self.mlps =nn.Sequential(nn.Conv2d(in_channels+3,256,kernel_size=1,bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(256,256,kernel_size=1,bias=False),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU()
                                   )



        self.output_mlp1 = nn.Sequential(nn.Conv2d(self.k, 256, kernel_size=[1,256],bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU() )

        self.output_mlp2 =nn.Sequential(nn.Conv1d(256,128,kernel_size=1,bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU())

        self.attention_unit =attention_unit(in_channels+3)
        self.ra_layer = RA_Layer1(in_channels + 3)
        self.up_ratio =up_ratio
    def forward(self,feature,input,rel_pos_emb):
        net = feature  # b,128,n*4
        B, C, N =net.shape

        #利用最远采样法进行近似均匀采样
        farthest=farthest_point_sample(net,int(N/self.up_ratio))
        point_cloud1 = net.reshape(B, N, C)
        downsample_points = index_points(point_cloud1, farthest).permute(0,2,1)   #b ,128,n

        grouped_points, grouped_features, _ = grouping(self.k, input, downsample_points, input)



        grouped_points -= torch.tile(input.unsqueeze(3), (1, 1, 1, self.k))

        features =torch.cat([grouped_features,grouped_points],dim=1)
        spatial_skip_connecttion, _ = torch.max(features, dim=3, keepdim=False)
        spatial_skip_connecttion =self.spatial_skip_mlp(spatial_skip_connecttion)

        features = self.mlps(features)
        grouped_features = features.transpose(1, 3)
        grouped_features = self.output_mlp1(grouped_features)
        grouped_features = grouped_features.squeeze(3)
        grouped_features = grouped_features + spatial_skip_connecttion
        grouped_features = self.output_mlp2(grouped_features)


        #提取全局信息
        global_feature =torch.cat([grouped_features,input],dim=1)

        global_feature =self.ra_layer(global_feature)
       # global_feature =self.attention_unit(global_feature)
        global_feature =self.mlp1(global_feature)  #有mlp-mlps

       # output=self.ra_layer(global_feature,rel_pos_emb)
        net = self.conv2(global_feature)
        return net
class down_block(nn.Module):
    def __init__(self, up_ratio=4, in_channels=128):
        super(down_block, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv1d(in_channels=in_channels*up_ratio, out_channels=in_channels, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.up_ratio = up_ratio

    def forward(self, inputs):
        net = inputs  # b,128,n*4
        net = net.reshape(net.shape[0],net.shape[1]*self.up_ratio,-1)#b,128,4,n
        net = self.conv(net)#b,256,1,n
        net = self.conv1(net)
        net = self.conv2(net)
        return net

def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max