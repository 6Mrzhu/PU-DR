import torch
import torch.nn as nn
from torch.nn import  Sequential as Seq
from network import up_projection_unit,mlp_conv,get_graph_feature
from network import LocalAttention,attention_unit,mlp
from knn_cuda import KNN
from pointutil.pointnet2_utils import gather_operation,grouping_operation
from torch_vertex import GraphConv2d
from torch_edge  import DilatedKnnGraph
from RA_layer import RA_Layer,RA_Layer1
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from network import up_projection_unit, mlp_conv, get_graph_feature
from network import LocalAttention, attention_unit, mlp
from knn_cuda import KNN
from pointutil.pointnet2_utils import gather_operation, grouping_operation
from RA_layer import RA_Layer, RA_Layer1
from torch_vertex import GraphConv2d
from torch_edge import DilatedKnnGraph


class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """

    def __init__(self, k=16):
        super(get_edge_feature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud):
        dist, idx = self.KNN(point_cloud, point_cloud)
        '''
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        idx = idx[:, 1:, :]
        point_cloud_neighbors = grouping_operation(point_cloud, idx.contiguous().int())
        point_cloud_central = point_cloud.unsqueeze(2).repeat(1, 1, self.k, 1)
        # print(point_cloud_central.shape,point_cloud_neighbors.shape)
        edge_feature = torch.cat([point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=1)

        return edge_feature, idx

        return dist, idx


class denseconv(nn.Module):
    def __init__(self, growth_rate=64, k=16, in_channels=6, isTrain=True):
        super(denseconv, self).__init__()
        self.edge_feature_model = get_edge_feature(k=k)
        '''
        input to conv1 is batch_size,2xn_dims,k,n_points
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=[1, 1]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=growth_rate + in_channels, out_channels=growth_rate, kernel_size=[1, 1]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=2 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=[1, 1]),
        )

    def forward(self, input):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
        y, idx = self.edge_feature_model(input)

        inter_result = torch.cat([self.conv1(y), y], dim=1)  # concat on feature dimension  #32+32
        inter_result = torch.cat([self.conv2(inter_result), inter_result], dim=1)  # 32+32+32
        final_result = torch.max(inter_result, dim=2)[0]  # pool the k channelaa
        return final_result, idx
        print(inter_result)

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.growth_rate = 32
        self.dense_n = 3
        self.knn = 16
        self.input_channel = 3
        comp = self.growth_rate
        '''
        make sure to permute the input, the feature dimension is in the second one.
        input of conv1 is batch_size,num_dims,num_points
        '''
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channel, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.denseconv1 = denseconv(in_channels=32 * 2, growth_rate=self.growth_rate,k=4)  # return batch_size,(3*32+32)=128,n,num_points

        self.denseconv2 = denseconv(in_channels=32 * 2, growth_rate=self.growth_rate, k=8)

        self.denseconv3 = denseconv(in_channels=32 * 2, growth_rate=self.growth_rate, k=16)

        self.rayler = RA_Layer1(416)

    def forward(self, input, rel_pos_emb):
        l0_features = self.conv1(input)  # b,32,n

        features_global = torch.max(l0_features, dim=2)[0]  ##global feature
        features_global = features_global.unsqueeze(2).repeat(1, 1, l0_features.shape[2])
        # print(l0_features.shape)

        l1_features, l1_index = self.denseconv1(l0_features)

        l2_features, l2_index = self.denseconv2(l0_features)

        l3_features, l3_index = self.denseconv3(l0_features)

        feature = torch.cat([features_global, l1_features, l2_features, l3_features], dim=1)
       # print(l1_features.shape)

        return feature  # 640


class Upsampler(nn.Module):
    def __init__(self, params=None):
        super(Upsampler, self).__init__()
        self.feature_extractor = feature_extraction()
        self.up_project_unit = up_projection_unit()

        self.conv1 = Seq(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = Seq(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
        )

    def forward(self, input):
        pos = input.permute(0, 2, 1)
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = torch.clamp(rel_pos.sum(-1), -5, 5)  
        feature = self.feature_extractor(input, rel_pos_emb)
        H = self.up_project_unit(feature, input, rel_pos_emb)
       # H = self.up_project_unit(feature)
        coord = self.conv1(H)
        coord = self.conv2(coord)
        return coord


class Coordinate_recon(nn.Module):
    def __init__(self, params):
        super(Coordinate_recon, self).__init__()
        self.feature_extractor = feature_extraction()
        self.up_ratio = params['up_ratio']
        self.num_point = params['patch_num_point']

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=648, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()

        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, input):
        feature = self.feature_extractor(input)
        coord = self.conv0(feature)
        coord = self.conv1(coord)
        coord = self.conv2(coord)
        return coord


class Discriminator(nn.Module):
    def __init__(self, params, in_channels):
        super(Discriminator, self).__init__()
        self.params = params
        self.start_number = 32
        self.bn3 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp_conv1 = mlp_conv(in_channels=in_channels, layer_dim=[self.start_number, self.start_number * 2])
        self.attention_unit = attention_unit(in_channels=self.start_number * 4)
        self.mlp_conv2 = mlp_conv(in_channels=self.start_number * 4,
                                  layer_dim=[self.start_number * 4, self.start_number * 8])
        self.mlp = mlp(in_channels=self.start_number * 8, layer_dim=[self.start_number * 8, 1])
        self.rayler = RA_Layer1(128)

    def forward(self, inputs):
        features = get_graph_feature(inputs, k=16)
        features = self.conv1(features)
        features = features.max(dim=-1, keepdim=False)[0]
        features_global = torch.max(features, dim=2)[0]  ##global feature
        z = features_global.unsqueeze(2).repeat(1, 1, features.shape[2])
        features = torch.cat([features, features_global.unsqueeze(2).repeat(1, 1, features.shape[2])], dim=1)
        # features = get_graph_feature(features, k=20)
        # features = self.conv1(features)
        # features = features.max(dim=-1, keepdim=False)[0]
        #features = self.attention_unit(features)
        features = self.rayler(features)
        features = self.mlp_conv2(features)
        features = torch.max(features, dim=2)[0]
        # print(features)
        output = self.mlp(features)

        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "up_ratio": 4,
        "patch_num_point": 100
    }
    generator = Upsampler(params).cuda()
    point_cloud = torch.rand(4, 3, 100).cuda()
    output = generator(point_cloud)
    print(output.shape)
    discriminator = Discriminator(params, in_channels=3).cuda()
    dis_output = discriminator(output)
    print(dis_output.shape)

    print(dis_output)
