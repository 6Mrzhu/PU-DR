import plyfile
import numpy as np
from open3d.cpu.pybind.io import read_point_cloud
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.io import read_point_cloud, write_point_cloud
from open3d.cpu.pybind.utility import Vector3dVector
class FarthestSamper:
    def __init__(self):
        pass
    def _cacl_distances(self,p0,points):
        return ((p0 -points)**2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts =np.zeros((k,3),dtype=np.float32)
        farthest_pts[0] =pts[np.random.randint(len(pts))]
        distance =self._cacl_distances(farthest_pts[0],pts)
        for i in range(1,k):
            farthest_pts[i] =pts[np.argmax(distance)]
            distance =np.minimum(
                distance ,self._cacl_distances(farthest_pts[i],pts)
            )
        return farthest_pts

def downsample_point(pts ,k):
    if pts.shape[0] >=2*k:
        sampler =FarthestSamper()
        return sampler(pts,k)
    else:
        #随机抽样np.random.choice（抽取样本数量，输出数量，replace为True可以取相同的数字）
        return pts[np.random.choice(pts.shape[0],k,replace=(k<pts.shape[0])),:]
def read_ply(file ,count=None):
    loaded =plyfile.PlyData.read(file)
    points =np.vstack([loaded['vertex'].data['x'],loaded['vertex'].data['y'],loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals =np.vstack([loaded['vertex'].data['nx'],loaded['vertex'].data['ny'],loaded['vertex'].data['nz']])
        points  =np.concatenate([points ,normals],axis=0)

    points =points.transpose(1,0)
    if count is not None:
        if count >points.shape[0]:
          # fill the point clouds with the random point
          tmp =np.zeros((count ,points.shape[1]),dtype=points.dtype)
          tmp[:points.shape[0] ,...] =points
          tmp[points.shape[0]:,...] =points[np.random.choice(points.shape[0],count-points.shape[0]),:]
          points =tmp
        elif count <points.shape[0]:
            points =downsample_point(points,count)

    return points
def read_pcd(filename ,count =None):
    points =read_point_cloud(filename)
    points =np.array(points.point).astype(np.float32)
    if count is not None:
        if count >points.shape[0]:
          # fill the point clouds with the random point
          tmp =np.zeros((count ,points.shape[1]),dtype=points.dtype)
          tmp[:points.shape[0] ,...] =points
          tmp[points.shape[0]:,...] =points[np.random.choice(points.shape[0],count-points.shape[0]),:]
          points =tmp
        elif count <points.shape[0]:
            points =downsample_point(points,count)
    return points
def load(filename ,count =None):
    if filename[-4:] ==".ply":
        points =read_ply(filename ,count)[:,:3]

    elif filename[-4:] ==".pcd":
        points =read_pcd(filename ,count)[:,:3]
    else:
        points =np.loadtxt(filename).astype(np.float32)
        if count is not None:
            if count > points.shape[0]:
                # fill the point clouds with the random point
                tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
                tmp[:points.shape[0], ...] = points
                tmp[points.shape[0]:, ...] = points[np.random.choice(points.shape[0], count - points.shape[0]), :]
                points = tmp
            elif count < points.shape[0]:
                points = downsample_point(points, count)
    return points