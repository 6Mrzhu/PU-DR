import torch.utils.data as data
import numpy as np
from torchvision import transforms
import os ,sys
import h5py
from data import data_util
from glob import glob
class PUNET_Dataset_Whole(data.Dataset):
   def __init__(self ,data_dir='../data/test_data/our_collected_data/MC_5k',n_input =1024):
      super().__init__()
      self.raw_input_points=5000
      self.n_input =1024

      file_list =os.listdir(data_dir)
      self.name =[x.split('.')[0] for x in file_list]
      self.sample_path =[os.path.join(data_dir,x)
                         for x in file_list]

   def __len__(self):
      return len(self.name)

   def __getitem__(self, index):
      points =np.loadtxt(self.sample_path[index])


      return  points




class PUNET_Dataset(data.Dataset):
    def __init__(self ,h5_file_path='E:\pythonma\PDN1\data\PUGAN_poisson_256_poisson_1024.h5',
                 split_dir='E:\pythonma\PDN1\data\ train_list.txt',
                 skip_rate =1 ,npoint =1024 ,use_random=True,
                 use_norm =True ,isTrain =True):
       super().__init__()

       self.isTrain =isTrain
       self.npoint =npoint
       self.use_random =use_random
       self.use_norm =use_norm

       h5_file =h5py.File(h5_file_path)
       self.gt =h5_file['poisson_1024'][:]
       self.input =h5_file['poisson_256'][:]
       assert len(self.input) ==len(self.gt) if use_random \
            else h5_file['montecarlo_1024'][:]
       self.data_npoint =self.input.shape[1]

       centroid =np.mean(self.gt[..., :3] ,axis=1 ,keepdims=True)
       furthest_distance =np.amax(np.sqrt(np.sum((self.gt[..., :3]-centroid)**2 ,axis=-1)),axis=1,keepdims=True)
       self.radius =furthest_distance[:,0]

       if use_norm:
          self.radius =np.ones(shape=(len(self.input)))
          self.gt[...,:3] -=centroid
          self.gt[...,:3] /=np.expand_dims(furthest_distance,axis=-1)
          self.input[...,:3] -=centroid
          self.input[...,:3] /=np.expand_dims(furthest_distance,axis=-1)

       self.split_dir =split_dir
       self.__load_split_file()

    def __load_split_file(self):
       index =np.loadtxt(self.split_dir)
       index =index.astype(np.int32)
       self.input =self.input[index]
       self.gt =self.gt[index]
       self.radius =self.radius[index]

    def __len__(self):
       return self.input.shape[0]

    def __getitem__(self, index):
       input_data =self.input[index]
       gt_data =self.gt[index]
       radius_data =self.radius[index]

       if not self.isTrain:
          return  input_data ,gt_data ,radius_data

       if self.use_norm:
          input_data,gt_data =data_util.rotate_point_cloud(input_data,gt_data)
          input_data,gt_data,scale =data_util.random_scale_point_cloud(input_data,gt_data,
                                                                 scale_low=0.9,scale_high=1.1)
          input_data,gt_data =data_util.shift_point_and_gt(input_data,gt_data,shift_range=0.1)
          radius_data =radius_data* scale

          #if np.random.rand() >0.5:
           #  input_data =data_util.jitter_perturbation_point_cloud(input_data,sigma=0.025 ,clip=0.05)
          #elif
           #  input_data =data_util.rotate_point_cloud(input_data,)
       else:
         raise NotImplementedError

       return input_data ,gt_data,radius_data
class Test_Dataset(data.Dataset):
   def __init__(self ,file_path ='D:\python\PDN1\input' ,npoints=256):
      super().__init__()

      file_list =os.listdir(file_path)
      self.names =[x.split('.')[0] for x in file_list]
      name =[x.split('.')[0] for x in file_list]
      print(name)
      file_path=os.path.abspath(file_path)
      paths =glob(os.path.join(file_path,'*.xyz'))
      print(len(paths))
      self.data =[]

      for path  in paths:
         points =np.loadtxt(path).astype(np.float32)[:,:3]
         self.data.append(points)
      self.npoints =npoints

   def __len__(self):
      return len(self.data)

   def __getitem__(self, item):
      x =self.data[item][:self.npoints,:]
      return x

if __name__ =="__main__":
   dataset = PUNET_Dataset()
   (input_data,gt_data,radius_data)=dataset.__getitem__(1)
   print(input_data.shape,gt_data.shape,radius_data.shape)
   print(radius_data)
   # dataset = Test_Dataset()
   # points=dataset.__getitem__(1)
   # print(points)
   # dst = Test_Dataset()
   # points, name = dst[0]
   # print(points, name)

