import scipy.io as sio
import os
# from scipy import signal
import math
import torch.nn as nn
from tools.spectral_tools import gen_mtf
from models.mlp import Gamma_net
from tools.psf2otf import *
from tools.coeffnet import *
# from coregistration import *
from models.myunet import skip
import torch.backends.cudnn as cudnn




# ###############################################
def upsample_interp23(image, ratio):
    # image = np.transpose(image, (2, 0, 1))

    b, r, c = image.shape

    CDF23 = 2 * np.array(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    d = CDF23[::-1]
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23

    first = 1
    for z in range(1, int(np.log2(ratio)) + 1):
        I1LRU = np.zeros((b, 2 ** z * r, 2 ** z * c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2] = image
            first = 0
        else:
            I1LRU[:, 0:I1LRU.shape[1]:2, 0:I1LRU.shape[2]:2] = image

        for ii in range(0, b):
            t = I1LRU[ii, :, :]
            for j in range(0, t.shape[0]):
                t[j, :] = ndimage.correlate(t[j, :], BaseCoeff, mode='wrap')
            for k in range(0, t.shape[1]):
                t[:, k] = ndimage.correlate(t[:, k], BaseCoeff, mode='wrap')
            I1LRU[ii, :, :] = t
        image = I1LRU

    # re_image = np.transpose(I1LRU, (1, 2, 0))
    re_image = I1LRU

    return re_image




class MyLoss(nn.Module):
    def __init__(self, mtf, ratio, device):
        super(MyLoss, self).__init__()

        # Parameters definition
        kernel = mtf
        self.pad_size = math.floor((kernel.shape[0] - 1) / 2)
        nbands = kernel.shape[-1]
        self.ratio = ratio
        self.device = device
        # Conversion of filters in Tensor
        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)
        kernel = torch.from_numpy(kernel).type(torch.float32)


        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.pad = nn.ReplicationPad2d(self.pad_size)

    def forward(self, outputs, r, c):
        x = self.pad(outputs)
        x = self.depthconv(x)
        x = x[:, :, 2::self.ratio, 2::self.ratio]

        return x

class BLU(nn.Module):
    def __init__(self, mtf, ratio, device):
        super(BLU, self).__init__()

        # Parameters definition
        kernel = mtf
        self.pad_size = math.floor((kernel.shape[0] - 1) / 2)
        nbands = kernel.shape[-1]
        self.ratio = ratio
        self.device = device
        # Conversion of filters in Tensor
        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.pad = nn.ReplicationPad2d(self.pad_size)

    def forward(self, outputs, r, c):
        x = self.pad(outputs)
        x = self.depthconv(x)
        xx = []
        for bs in range(x.shape[0]):
            xx.append(fineshift(torch.unsqueeze(x[bs], 0), r[bs], c[bs], self.device))
        x = torch.cat(xx, 0)
        # x = x[:, :, 2::self.ratio, 2::self.ratio]

        return x.type(torch.float64)








def x_sub(fp,otf,m,theta,eta,lambda_1):
    x = np.zeros(fp.shape)
    for i in range(fp.shape[0]):
        up=2*lambda_1*np.fft.fft2(fp[i,:,:].squeeze(), axes=(0, 1))+eta*np.fft.fft2(m[i,:,:].squeeze(), axes=(0, 1))*(otf[i,:,:].conj())\
        -np.fft.fft2(theta[i,:,:].squeeze(), axes=(0, 1))*(otf[i,:,:].conj())
        down=2*lambda_1+eta*otf[i,:,:]*(otf[i,:,:].conj())
        FS=up/down
        x[i,:,:] = np.abs(np.fft.ifft2(FS, axes=(0, 1)))
    return x

def m_sub(yst, xb, sst, theta, eta):
    up = 2*yst+eta*xb+theta
    down = 2*sst+eta
    m = up/down
    return m



def test_dip(lrms,pan,gt,sensor):
    # ================== Pre-Define =================== #
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cudnn.benchmark = True  ###自动寻找最优算法
    torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    dtype = torch.FloatTensor
    # dtype = torch.DoubleTensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ms_np = lrms.numpy()
    pan_np = pan.squeeze().numpy()
    gt_np = gt.numpy()
    u_hs = upsample_interp23(ms_np, 4)
    p_3d_np = np.tile(pan_np, (u_hs.shape[0], 1, 1))
    p_3d_np = np.clip(p_3d_np, 0, 1)



    ########=============================#################
    ratio = 4
    kernel = gen_mtf(ratio, sensor)  # 4x41x41
    ###############################################
    kernel = kernel.transpose(2, 0, 1)
    input_depth = p_3d_np.shape[0]


    # ====================================
    ############  parameters  ############
    # ====================================
    lam, mu = 100,0.001
    net_iter = 8000
    LR = 0.1



    # loss and model
    downgrade = MyLoss(gen_mtf(ratio, sensor), ratio, device).to(device)

    att_pan_model = Gamma_net(u_hs.shape[0]).to(device)


    ############## network parameters ###############
    

    pad = 'reflection'


    tt = 4

    model = skip(input_depth, input_depth,
                num_channels_down=[128] * tt,
                num_channels_up=[128] * tt,
                num_channels_skip=[4] * tt,
                filter_size_up=3, filter_size_down=3, filter_skip_size=1,
                upsample_mode='bilinear',
                need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype).to(device)





    p =  [x for x in att_pan_model.parameters()]
    p = p + [x for x in model.parameters()]
    optimizer = torch.optim.Adam(p, lr=LR)


    loss_fn = torch.nn.MSELoss().type(dtype).to(device)

    batch_sz = 4
    r = torch.tensor([0])
    c = torch.tensor([0])
    r = r.repeat(batch_sz, 1)
    c = c.repeat(batch_sz, 1)

    # =============================================
    lrms_torch = lrms[None, :].to(device)
    p_3d_torch = torch.from_numpy(p_3d_np[None, :]).type(dtype).to(device)



    pan_torch = pan.to(device)
    pan_torch_expand = pan_torch.repeat(input_depth, 1, 1)
    pan_torch_expand = pan_torch_expand[None, :]




    fp, psnr_list = coeffnet(gt_np,att_pan_model, p_3d_torch ,pan_torch_expand ,lrms_torch, model, optimizer, loss_fn, downgrade, net_iter, lam, mu,r,c)
    x = (fp * p_3d_torch).detach().cpu().numpy().squeeze()
    x = np.clip(x, 0, 1)
    gt_torch = torch.from_numpy(gt_np).type(torch.float32)
    coff_gt = gt_torch/p_3d_torch.cpu()


    #============== VO ==================================
    batch_sz = 4
    r = torch.tensor([0])
    b = torch.tensor([0])
    r = r.repeat(batch_sz, 1)
    b = b.repeat(batch_sz, 1)

    fp = (fp * p_3d_torch).cpu().numpy().squeeze()
    c=0
    BLUR=BLU(gen_mtf(ratio, sensor),ratio,device).type(dtype).to(device)
    for q in {1}:#range(1,4): please adjust the parameter for better perofrmance
        for w in {2}:#range(1,6): please adjust the parameter for better perofrmance
            lambda_1 = 10**(-q)
            eta = 10**(-w)

            theta = np.zeros(p_3d_np.shape)
            x = np.zeros(p_3d_np.shape)
            m = np.zeros(p_3d_np.shape)
            otf = np.zeros(m.shape)
            for k in range(m.shape[0]):
                otf[k,:,:] = psf2otf(kernel[k,:,:], [p_3d_np.shape[1],p_3d_np.shape[2]])
            sst = np.zeros(m.shape)
            yst = np.zeros(m.shape)
            for i in range(m.shape[0]):
                sst[i, 2::4, 2::4] = np.ones([lrms.shape[1], lrms.shape[2]])
                yst[i, 2::4, 2::4] = lrms[i, :, :].squeeze()
            for j in range(100):
                x = x_sub(fp,otf,m,theta.squeeze(),eta,lambda_1)
                x_torch=torch.from_numpy(x[None, :]).type(dtype).to(device)
                xb = BLUR(x_torch,r,b)
                xb = xb.cpu().numpy().squeeze()
                x = np.clip(x, 0, 1)
                m = m_sub(yst, xb, sst, theta.squeeze(), eta)
                theta = theta+eta*(xb-m)
            a, psnr_x = rmse1(x, gt)
            a1 = x.transpose(1,2,0)
            a2 = gt.numpy().transpose(1,2,0)
            ssim_x = ssim(np.uint8(np.round(a1* 255)),np.uint8(np.round(a2* 255)))
            print('PSNR：{:.3f} SSIM：{:.3f} Lambda:{:.5f} eta::{:.5f} '.format(psnr_x,ssim_x,lambda_1,eta))

    return x, psnr_x









if __name__ == '__main__':

    file_path = "matlab.mat"
    data = sio.loadmat(file_path)
    used_ms = data['lrms'].transpose(2,0,1)  # 64x64x4
    GT = data['gt'].transpose(2,0,1)   # 256x256x4
    pan = data['pan']   # 256x256
    used_pan = np.expand_dims(pan, axis=2)
    used_pan =used_pan.transpose(2,0,1)
    dtype=torch.FloatTensor

    pan = torch.from_numpy(used_pan).type(dtype)
    lrms = torch.from_numpy(used_ms).type(dtype)
    gt = torch.from_numpy(GT).type(dtype)
    sensor="none"
    sr=test_dip(lrms,pan,gt,sensor)





