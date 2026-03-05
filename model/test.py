import argparse
import os ,sys
sys.path.append("../")

parser =argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu',type=int ,default=0 ,help='use to GPU',required=False)
parser.add_argument('--resume',type=str,default='../checkpoints/the_project_name/G_iter_99.pth', required=False)
parser.add_argument('--exp_name',type=str ,default='exp',required=False)

args =parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
from torch.utils.data import DataLoader
from util1.xyz_util import save_xyz_file
from model import Upsampler
#from Unsampling.model import Upsampler
from data_loader import Test_Dataset,Test_Dataset1,Test_noise,KITTI
from train_option import get_train_options
def index_point(points ,idx):
    raw_size =idx.size()
    idx =idx.reshape(raw_size[0],-1)
    #torch.gather(input, dim, index, out=None)

    res =torch.gather(points,1,idx[...,None].expand(-1,-1,points.size(-1)))
    return res.reshape(*raw_size ,-1)

if __name__ =='__main__':
   #  params = get_train_options()

   #  params["patch_num_point"] = 1024
   #  params['up_ratio'] = 4
   #  model =Upsampler()
   #  args.resume ='../pdn1/checkpoints/the_project_name/G_iter_99.pth'
    model = Upsampler()
    args.resume = '../pdn1/checkpoints1/the_project_name/G_iter_99.pth'#30 50
    if os.path.exists( args.resume)==True:
        print(args.resume)
        print(True)
        print('路径存在')
    else:
        print('路径不存在')
        print(False)
    checkpoint =torch.load(args.resume)
    #load_state_dict
    model.load_state_dict(state_dict=checkpoint)
    #model.eval()
    model.eval().cuda()
   # eval_dat =PUNET_Dataset(h5_file_path='../Patches_noHole_and_collected.h5', split_dir=param['test_split'],
    #                         isTrain=False)
    #eval_dat =Test_Dataset(npoints=256)
    #eval_dat = Test_Dataset1(npoints=2048)
    eval_dat = KITTI(npoints=119862)
    #DataLoader
    eval_loader =DataLoader(eval_dat,batch_size=1,shuffle=False ,pin_memory=True,num_workers=0)

    name=eval_dat.names
    exp_name =args.exp_name
    save_dir =os.path.join('../outputs',exp_name)

    if os.path.exists(save_dir) ==False:
        os.makedirs(save_dir)

    print("initialize")

    with torch.no_grad():
        for itr ,batch in enumerate(eval_loader):
            print(itr)
            nam =name[itr]
            print(name)
            points =batch[:,:,0:3].permute(0,2,1).float().cuda()
            preds=model(points)

            preds =preds.permute(0,2,1).data.cpu().numpy()[0]
            points =points.permute(0,2,1).data.cpu().numpy()
            save_file = '../outputs/img4096/{}/{}.xyz'.format(exp_name, nam)
          #  save_file = '../outputs/{}/{}.xyz'.format(exp_name, nam)

            save_xyz_file(preds,save_file)


    print("success")
