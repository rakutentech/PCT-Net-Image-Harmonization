import torch
import numpy as np
import kornia as K
import scipy

def pre_process(lum, gamma=2.2):

    lum_flat= lum.flatten()                                                  
    lum_sorted = torch.argsort(lum_flat)                                       
    lum_cut_pos = int(lum_flat.size(0)*0.05)                                   
    lmin = lum_flat[lum_sorted[lum_cut_pos]]                              
    lmax = lum_flat[lum_sorted[lum_flat.size(0)-lum_cut_pos]]
    lum = torch.clip(lum, lmin, lmax)
    lum = (lum - lmin) / (lmax - lmin) * 100 

    return lum

def post_process(img):
    img = K.color.lab_to_rgb(img)
    img = torch.pow(img, 1/2.2)
    
    return img


# Chrominance transfer
def transfer_chrom(inp, src):
    '''
    inp (3, H, W):              input image
    src (3, H, W):              source image

    out (3, H, W):              output image
    '''
    
    m_i = torch.mean(inp, dim=(1,2))
    m_s = torch.mean(src, dim=(1,2))
    s_i = torch.maximum(torch.cov(inp.view(3, -1)), 7.5*torch.eye(3).to(inp.get_device())) 
    s_s = torch.maximum(torch.cov(src.view(3, -1)), 7.5*torch.eye(3).to(inp.get_device()))

    w_i, v_i = torch.linalg.eigh(s_i)
    w_i[w_i<0] = 0
    da = torch.diag(torch.sqrt(w_i + 1e-10))
    c = torch.mm(da, torch.mm(v_i.T, torch.mm(s_s, torch.mm(v_i, da))))
    w_c, v_c = torch.linalg.eigh(c)
    w_c[w_c<0] = 0
    dc = torch.diag(torch.sqrt(w_c + 1e-10))
    da_inv = torch.diag(1/(torch.diag(da)))
    T = torch.mm(v_i, torch.mm(da_inv, torch.mm(v_c, torch.mm(dc, torch.mm(v_c.T, torch.mm(da_inv, v_i.T))))))

    out = torch.matmul(T, (inp.view(3, -1) - m_i[:,None])) + m_s[:,None]

    return out.view(inp.size())

def m_transfer_chrom(inp, src):
    '''
    inp (3, M):              input image
    src (3, M):              source image

    out (3, M):              output image
    '''
    
    m_i = torch.mean(inp, dim=1)
    m_s = torch.mean(src, dim=1)
    s_i = torch.maximum(torch.cov(inp), 7.5*torch.eye(3).to(inp.get_device())) 
    s_s = torch.maximum(torch.cov(src), 7.5*torch.eye(3).to(inp.get_device()))

    w_i, v_i = torch.linalg.eigh(s_i)
    w_i[w_i<0] = 0
    da = torch.diag(torch.sqrt(w_i + 1e-10))
    c = torch.mm(da, torch.mm(v_i.T, torch.mm(s_s, torch.mm(v_i, da))))
    w_c, v_c = torch.linalg.eigh(c)
    w_c[w_c<0] = 0
    dc = torch.diag(torch.sqrt(w_c + 1e-10))
    da_inv = torch.diag(1/(torch.diag(da)))
    T = torch.mm(v_i, torch.mm(da_inv, torch.mm(v_c, torch.mm(dc, torch.mm(v_c.T, torch.mm(da_inv, v_i.T))))))

    out = torch.matmul(T, (inp - m_i[:,None])) + m_s[:,None]

    return out

# Luminance transfer
def hist_sampler(img, num_samples, bins=256):

    xs = torch.linspace(img.min(), img.max(), bins+1).to(img.get_device())
    ys = torch.histc(img, bins=bins, min=img.min(), max=img.max())
    
    ys = ys/torch.sum(ys)
    ys = torch.cumsum(ys, 0)
    percents = torch.arange(1, 1+num_samples).to(img.get_device()) / num_samples
    index = torch.searchsorted(ys, percents)

    return xs[index]

def transfer_func(param, lum_i):
    tmp = torch.arctan(param[0]/param[1])
    return (tmp + torch.arctan((lum_i-param[0])/param[1])) / (tmp + torch.arctan((1-param[0])/param[1])) 

def np_transfer_func(param, lum_i):
    tmp = np.arctan(param[0]/param[1])
    return (tmp + np.arctan((lum_i-param[0])/param[1])) / (tmp + np.arctan( (1-param[0])/param[1] ))

def transfer_lum(inp, src, num_samples=32, tau=0.4):
    '''
    inp (H, W):                 luminance of input image
    src (H, W):                 luminance of source image

    out (H, W):                 luminance of output image
    '''

    lum_i = hist_sampler(inp/(inp.max()+1e-10), num_samples)
    lum_s = hist_sampler(src/(src.max()+1e-10), num_samples)

    lum_cal = lum_i + (lum_s - lum_i) * (tau / min(tau, torch.linalg.vector_norm(lum_s-lum_i, ord=float('inf'))))
    target_function = lambda param: np.linalg.norm(np_transfer_func(param, lum_i.cpu().numpy())-lum_cal.cpu().numpy()) ** 2
    result = scipy.optimize.minimize(target_function, np.random.random_sample([2]), method = 'Nelder-Mead', bounds=((0,1), (1e-10, 1)), options={'disp': False})
    
    solution = torch.tensor(result.x, device=inp.get_device())

    out = transfer_func(solution, inp/inp.max())
    out = torch.clip(out, 0) * inp.max()

    return  out