# -*- coding:utf-8 -*-
# Created by Jin-Liang Xiao 2025-08-26

import h5py
from tools.dipprocess import *
import time
###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################

def load_set(file_path):
    data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

    # tensor type:
    lms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    print(lms1.shape)
    lms1 = np.array(lms1, dtype=np.float32) /1023.0# 2047.0
    lms = torch.from_numpy(lms1)  # NxCxHxW  or HxWxC
    print(lms.shape)

    pan1 = data['pan'][...]   # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 1023.0# 2047.0
    pan = torch.from_numpy(pan1)
    print(pan.shape)

    gt = data['gt'][...]  # NxCxHxW = 4x8x512x512
    print(gt.shape)
    gt = np.array(gt, dtype=np.float32) /1023.0# 2047.0
    gt = torch.from_numpy(gt)  # NxCxHxW  or HxWxC
    print(gt.shape)

    return lms, pan, gt


# ==============  Main test  ================== #

def test(file_path):
    lms, pan, test_gt = load_set(file_path)
    sensor='none'
    x1, x2 = lms, pan   # read data: CxHxW (numpy type)
    num_exm = x1.shape[0]
    results=np.zeros(test_gt.permute(0,2,3,1).shape)
    psnr_list=np.zeros([20,1])
    time_start = time.time()
    for index in range(num_exm):  # save the LightNet results for matlab evaluate code
        file_name = "proposed_reduced_gf2.mat"
        directory_name = "./results/GF/"
        print('data index={:.3f}'.format(index))
        x1_1=x1[index, :, :, :]
        x2_1=x2[index, :, :, :]
        gt = test_gt[index, :, :, :]
        sr, psnr_ = test_dip(x1_1,x2_1,gt,sensor)
        results[index, :, :, :]=sr.transpose(1,2,0)
        psnr_list[index, :] = psnr_

    time_end = time.time()
    time_total = time_end - time_start
    gt_trans = test_gt.permute(0,2,3,1).cpu().numpy() 
    lrms_trans = lms.permute(0,2,3,1).cpu().numpy() 
    pan_trans = pan.cpu().numpy().squeeze()   
    psnr_mean = np.mean(psnr_list)
    print('Best average PSNR_x={:.3f}'.format(psnr_mean))
    print('Total Time ={:.3f}'.format(time_total))


    # save_name = os.path.join(directory_name, file_name)
    # sio.savemat(save_name, {'proposed': results, 'lrms': lrms_trans, 'pan': pan_trans,'gt': gt_trans})

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':

    file_path = "./dataset/test_gf2_multiExm1.h5"
    test(file_path)