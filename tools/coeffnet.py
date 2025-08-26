import torch
from scipy import signal
from tools.metrics import *




def grad_loss2(img):
    """
    Compute total variation loss.

    """
    b, chan, height, width = img.size()
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = (h_variance + w_variance)/ height / width / chan
    return loss


def Gaussian_downsample(x, psf, s):
    y = np.zeros((x.shape[1], int(x.shape[2] / s), int(x.shape[3] / s)))
    x = x.squeeze()
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='wrap', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    y = np.expand_dims(y, axis=0)
    return y





def coeffnet(gt,att_pan_model, pan, pan_torch_expand, lrms, model, optimizer, loss_fn, downgrade, net_iter,
        lam, mu, r, c):
    psnr_list = []
    for i in range(net_iter):

        optimizer.zero_grad()
        model_out = model(pan_torch_expand)
        pan_out = att_pan_model(pan_torch_expand)/pan
        downgraded_shifted_outputs = downgrade(model_out* pan, r, c)
        y_loss = loss_fn(downgraded_shifted_outputs, lrms)
        z_loss = grad_loss2(pan_out-model_out)
        loss =  z_loss * mu+y_loss * lam # + 0.1 * g_loss
        loss.backward()
        optimizer.step()
        if (i) % 1000 == 0:

            a, b = rmse1(np.clip((model_out * pan).detach().cpu().numpy().squeeze(), 0, 1), gt)
            print('net_iter {}, y_loss:{:.7f}, g_loss:{:.7f}, z_loss:{:.7f}, PSNR:{:.3f}'.format(i,
                                                                                                 y_loss.detach().cpu().numpy(),
                                                                                                 0,
                                                                                                 z_loss.detach().cpu().numpy(),
                                                                                                 b))
            psnr_list.append(b)
    return model_out.detach(), psnr_list