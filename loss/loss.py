import math

import torch
import torch.nn as nn
from knn_cuda import KNN
import chamfer_distance
import pointutil.pointnet2_utils as pn2_utils
from loss.earth_movers_distance.emd import EarthMoverDistance
emd = EarthMoverDistance()

def knn_point(group_size,point_cloud ,query_cloud,transpose_model =True):
    knn_obj =KNN(k=group_size ,transpose_mode=True)
    dist ,idx= knn_obj(point_cloud,query_cloud)
    return dist ,idx
class Loss(nn.Module):
    def __init__(self ,radius=1):#之前为1
        super(Loss, self).__init__()
        self.radius =radius
        self.knn_repulsion =KNN(k=20 ,transpose_mode=True)
        self.knn_uniform =KNN(k=2,transpose_mode=True)

    def get_emd_loss(self,pred,gt,pcd_radius):
        '''
        pred and gt is B N 3
        '''
        dist = emd(pred, gt)
        dist /= pcd_radius
        return torch.mean(dist)
    def get_repulsion_loss(self,pcd ,h=0.0005):
        dist ,idx =KNN(pcd ,pcd) #B N K

        dist =dist[:,:,1:5]**2 #top 4 cloest neighbors

        loss =torch.clamp(-dist+h ,min=0)
        loss =torch.mean(loss)

        return loss
    def get_cd_loss(self,pred ,gt ,pcd_radius=1):
        cost_for ,cost_bac =chamfer_distance.chamfer_distance(gt,pred)
        cost =0.8 *cost_for +0.2*cost_bac
        cost /=pcd_radius
        cost =torch.mean(cost)
        return cost
    def get_uniform_loss(self,pcd,percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
        B ,N,C =pcd.shape[0] ,pcd.shape[1] ,pcd.shape[2]
        npoint =int(N*0.05)
        loss =0
        further_point_idx =pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz =pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx) #B,C,N
        for p in percentage:
            nsample =int(N*p)
            r =math.sqrt(p*radius)
            disk_area =math.pi*(radius**2)/N
            idx =pn2_utils.ball_query(r,nsample,pcd.contiguous(),new_xyz.permute(0,2,1).contiguous())
            expect_len =math.sqrt(disk_area)
            grouped_pcd =pn2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx)
            grouped_pcd =grouped_pcd.permute(0,2,3,1) #B N nsample C
            grouped_pcd =torch.cat(torch.unbind(grouped_pcd,dim=1),dim=1) #B*N nsample C

            dist,_ =self.knn_uniform(grouped_pcd,grouped_pcd)
            uniform_dist = dist[:, :, 1:]  # B*N nsample 1
            uniform_dist = torch.abs(uniform_dist + 1e-8)
            uniform_dist = torch.mean(uniform_dist, dim=1)
            uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + 1e-8)
            mean_loss = torch.mean(uniform_dist)
            mean_loss = mean_loss * math.pow(p * 100, 2)
            loss += mean_loss
        return loss / len(percentage)


if __name__ =="__main__":
    loss =Loss().cpu()
    point_cloud =torch.rand(4,128,3).cpu()
    gt =torch.randn(4,128,3).cpu()
    p =loss.get_cd_loss(point_cloud,gt)
    print(p)
