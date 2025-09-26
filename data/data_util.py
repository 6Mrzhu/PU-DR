import numpy as np

def nonuniform_sampling(num ,sample_num):
    sample =set()
    loc =np.random.rand()*0.8 +0.1
    while len(sample) < sample_num:
        a =int(np.random.normal(loc=loc,scale=0.3)*num) #高斯分布的概率密度随机数
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)

def rotate_point_cloud(input_data ,gt_data =None):
    #Randomly rotate the point clouds

    angles =np.random.uniform(size=(3))*2*np.pi #生成大小为3的随机数
    Rx =np.array([
        [1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]),-np.cos(angles[0])]])

    Ry =np.array([
        [np.cos(angles[1]) ,0 ,-np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]) ,0 ,np.cos(angles[1])]])
    Rz =np.array([
        [np.cos(angles[2]) ,-np.sin(angles[2]),0],
        [-np.sin(angles[2]) ,np.cos(angles[2]),0],
        [0,0,1]])

    rotation_matrix =np.dot(Rz,np.dot(Rx,Ry))

    input_data[:,:3] =np.dot(input_data[:,:3],rotation_matrix)
    if input_data.shape[1] >3: #a.shape[1]查看列数，[0]查看行数
        input_data[:,3:] =np.dot(input_data[:,3:],rotation_matrix)

    if gt_data is not None:
        gt_data[:,:3] =np.dot(gt_data[:,:3] ,rotation_matrix)
        if gt_data.shape[1] >3:
            gt_data[:,3:] =np.dot(gt_data[:,3:],rotation_matrix)

    return input_data ,gt_data

#任意缩放点云
def random_scale_point_cloud(input_data ,gt_data ,scale_low=0.5 ,scale_high=2):
    scale =np.random.uniform(scale_low ,scale_high)#进行随机采样

    input_data[:,:3]*=scale
    if gt_data is not None:
        gt_data[:,:3] *=scale

    return input_data ,gt_data ,scale

#任意抖动点云

def jitter_perturbation_point_cloud(input_data ,sigma=0.005, clip=0.02):
    assert (clip>0)
    jitter =np.clip(sigma*np.random.randn(*input_data.shape),-1*clip ,clip)
    jitter[:,:3] =0
    input_data+=jitter
    return input_data

#对点云数据进行移动
def shift_point_and_gt(input_data,gt_data=None,shift_range=0.3):
    shifts =np.random.uniform(-shift_range,shift_range ,3)

    input_data[:,:3] +=shifts
    if gt_data is not None:
        gt_data[:,:3] +=shifts
    return input_data,gt_data
#对点云进行归一化操作
def normalize_point_cloud(input):
    """

    :param input: pc [N ,P,3]
    :return: pc centroid ,furthest_distance
    """
    if len(input.shape) ==2:
        axis =0
    elif len(input.shape)==3:
        axis =1
    centroid =np.mean(input,axis=axis,keepdims=True)
    input =input -centroid
    furthest_distance =np.max(
        np.sqrt(np.sum(input **2, axis=-1 ,keepdims=True)),axis=axis,keepdims=True)
    input =input /furthest_distance
    return input ,centroid ,furthest_distance

if __name__ == "__main__":
    import torch
    point_cloud = torch.rand(4, 3, 100)
    point_cloud =np.array(point_cloud)
    input ,centroid ,furthest =normalize_point_cloud(point_cloud)
    print(input.shape)
    print("-------------------------")
    print(centroid)
    print(furthest)