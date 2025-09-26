import torch
import torch.nn as nn
from torch.nn import  Sequential as Seq
from torch_nn import BasicConv, batched_index_select
from torch_edge import DilatedKnnGraph,DenseDilated
from torch_edge import pairwise_distance
import torch.nn.functional as F
def EdgeConv2d(x, edge_index):
    x_i = batched_index_select(x, edge_index[1])
    x_j = batched_index_select(x, edge_index[0])
    max_value = torch.cat([x_i, x_j - x_i], dim=1)
    return max_value

class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels):
        super(GraphConv2d, self).__init__()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, edge_index):
        x = EdgeConv2d(x, edge_index)
        x = self.conv1(x)
        x, _ = torch.max(x, -1, keepdim=False)
        return x
class GraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.gconv = GraphConv2d(in_channels, out_channels)

    def forward(self, x, edge_index, pos=None):
        if self._needs_pos:
            return self.gconv(x, pos=pos, edge_index=edge_index)
        else:
            return self.gconv(x, edge_index=edge_index)
def dense_knn_matrix(x, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = torch.arange(0, n_points, device=x.device).expand(batch_size, k, -1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)
class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = dense_knn_matrix

    def forward(self, x):
        edge_index = self.knn(x, self.k )
        return self._dilated(edge_index)
class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels)
        self.k = kernel_size
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)
class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64,  kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None,bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1)
class DenseGCN(nn.Module):
    def __init__(self,in_channels=3,channels=64,norm=None,n_blocks=3, bias=True, stochastic=False, epsilon=0.0, k=9, c_growth=64,knn='matrix',
                 act='relu',conv='edge'):
        super(DenseGCN, self).__init__()
        self.bn1 = nn.BatchNorm2d(32)
        self.n_blocks =n_blocks
        #密集连接 每一层已做完拼接
        self.backbone = Seq(*[DenseDynBlock2d(channels + c_growth * i, c_growth, k, 1 + i, conv, act,
                                              norm, bias, stochastic, epsilon, knn)
                              for i in range(self.n_blocks - 1)])

        fusion_dims = int(
            (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
       # self.fusion_block = BasicConv([fusion_dims, 128], 'leakyrelu', norm, bias=False) #融合特征的
        self.conv1 = nn.Sequential(nn.Conv2d(fusion_dims,32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5))#使用1×1的卷积
        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(in_channels, channels)
    def forward(self, x, edge_index=None):
        feats = self.head(x, self.knn(x[:, 0:3]))
        feat1 =[self.head(x, self.knn(x[:, 0:3]))]
      #  feat =[]
        for i in range(self.n_blocks - 1):
            w1=feat1[-1]
       #     p =self.backbone[i](feats)
            z=self.backbone[i]
            p1 =self.backbone[i](feat1[-1])
            feat1.append(self.backbone[i](feat1[-1]))
         #   feat.append(p)
         #   feats =p
        feats = torch.cat(feat1, dim=1)
        P = torch.cat(feat1, dim=1).unsqueeze(-1) #累加融合 可以尝试平均融合
        fusion = self.conv1(P).squeeze(-1)  #特征降维
        return fusion


class DenseGCN1(nn.Module):
    def __init__(self,channels=3,norm=None, bias=False, stochastic=False, epsilon=0.0, k=9, c_growth=32,knn='matrix',
                 act='relu',conv='edge'):
        super(DenseGCN1, self).__init__()
        self.bn1 = nn.BatchNorm2d(32)
        #使用不扩展的的密集图卷积
        self.Dense1 =DynConv2d(channels, c_growth, k, 1, conv, act,
                                              norm, bias, stochastic, epsilon, knn)
        self.Dense2 =DynConv2d(channels+c_growth ,c_growth,k, 1, conv, act,
                                              norm, bias, stochastic, epsilon, knn)
        self.Dense3 =DynConv2d(channels+c_growth*2 ,c_growth,k,1, conv, act,
                                              norm, bias, stochastic, epsilon, knn)
    def forward(self, x, edge_index=None):
        inter_result =torch.cat([self.Dense1(x),x],dim=1)
        inter_result =torch.cat([self.Dense2(inter_result),inter_result],dim=1)
        inter_result =torch.cat([self.Dense3(inter_result),inter_result],dim=1)
        inter_result =inter_result
        return  inter_result


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #  params = {
  #       "up_ratio": 4,
  #       "patch_num_point": 100
   # }
    generator =DenseGCN1().cuda()
    point_cloud = torch.rand(4, 3, 100).cuda()
    output = generator(point_cloud)
    print(output.shape)