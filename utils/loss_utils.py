#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def lpips_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

    

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def loss_depth_grad(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = img_grad_x / (torch.abs(img_grad_x) + 1e-6)
    weight_y = img_grad_y / (torch.abs(img_grad_y) + 1e-6)

    depth_grad_x = depth[:, :, :, :-1] - depth[:, :, :, 1:]
    depth_grad_y = depth[:, :, :-1, :] - depth[:, :, 1:, :]
    grad_x = depth_grad_x / (torch.abs(depth_grad_x) + 1e-6)
    grad_y = depth_grad_y / (torch.abs(depth_grad_y) + 1e-6)

    loss = l1_loss(grad_x, weight_x) + l1_loss(grad_y, weight_y)
    return loss

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def margin_l1_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask].abs()).mean()
    else:
        return ((network_output - gt)[mask].abs()).mean(), mask
    

def kl_loss(input, target):
    input = F.log_softmax(input, dim=-1)
    target = F.softmax(target, dim=-1)
    return F.kl_div(input, target, reduction="batchmean")


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def lpr_loss2(gsplat_render_maps, point_render_maps, device):
    total_loss = 0.0
    for gsplat_render, point_render in zip(gsplat_render_maps, point_render_maps):
        point_render = torch.tensor(point_render, dtype=torch.float32).to(device)

        # Calculate the mean squared error between the images
        loss = torch.abs((gsplat_render - point_render)).mean()
        total_loss += loss

    # Calculate the average loss across all images
    avg_loss = total_loss / len(gsplat_render_maps)

    return avg_loss

def lpr_loss(gsplat_render, point_render, device):
    if not isinstance(point_render, torch.Tensor):
        point_render = torch.tensor(point_render, dtype=torch.float32).to(device)

    loss = torch.abs((gsplat_render - point_render)).mean()

    return loss

def pearson_depth_loss(depth_src, depth_target):
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
    num_box_h = depth_src.shape[0] // box_p
    num_box_w = depth_src.shape[1] // box_p
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p
    n_corr = int(p_corr * num_box_h * num_box_w)

    x_0 = torch.randint(0, max_h, (n_corr,), device='cuda')
    y_0 = torch.randint(0, max_w, (n_corr,), device='cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p

    _loss = sum(
        pearson_depth_loss(
            depth_src[x0:x1, y0:y1].reshape(-1),
            depth_target[x0:x1, y0:y1].reshape(-1)
        )
        for x0, x1, y0, y1 in zip(x_0, x_1, y_0, y_1)
    )

    return _loss / n_corr

###输入是四维tensor
class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 0.1

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(1, 3, input_im.shape[2], input_im.shape[3])
        return out

    def norm(self, tensor, p):
        return torch.mean(torch.pow(torch.abs(tensor), p))

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        # print(self.input.shape)
        sigma_color = -1.0 / 2 * self.sigma * self.sigma
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * self.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p)
        pixel_grad2 = w2 * self.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p)
        pixel_grad3 = w3 * self.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p)
        pixel_grad4 = w4 * self.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p)
        pixel_grad5 = w5 * self.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p)
        pixel_grad6 = w6 * self.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p)
        pixel_grad7 = w7 * self.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p)
        pixel_grad8 = w8 * self.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p)
        pixel_grad9 = w9 * self.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p)
        pixel_grad10 = w10 * self.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p)
        pixel_grad11 = w11 * self.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p)
        pixel_grad12 = w12 * self.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p)
        pixel_grad13 = w13 * self.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p)
        pixel_grad14 = w14 * self.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p)
        pixel_grad15 = w15 * self.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p)
        pixel_grad16 = w16 * self.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p)
        pixel_grad17 = w17 * self.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p)
        pixel_grad18 = w18 * self.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p)
        pixel_grad19 = w19 * self.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p)
        pixel_grad20 = w20 * self.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p)
        pixel_grad21 = w21 * self.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p)
        pixel_grad22 = w22 * self.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p)
        pixel_grad23 = w23 * self.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p)
        pixel_grad24 = w24 * self.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term

# ============================================================
# 3-Branch Loss Functions: Color / Texture / Denoiser
# ============================================================

def rgb_to_y(img):
    """Convert RGB (C,H,W) to luminance (1,H,W)."""
    return (0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3])

def local_contrast_normalize(x, kernel_size=15, eps=1e-5):
    """Local Contrast Normalization: (x - local_mean) / (local_std + eps).
    Input: (1,C,H,W) or (C,H,W). Output: same shape."""
    needs_batch = (x.dim() == 3)
    if needs_batch:
        x = x.unsqueeze(0)
    pad = kernel_size // 2
    # Gaussian-like averaging via avg_pool
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    var = F.avg_pool2d(x ** 2, kernel_size, stride=1, padding=pad) - mean ** 2
    std = torch.sqrt(var.clamp(min=0) + eps)
    out = (x - mean) / std
    if needs_batch:
        out = out.squeeze(0)
    return out

def laplacian_pyramid_loss(pred, target, levels=3):
    """Multi-scale Laplacian pyramid L1 loss.
    Input: (C,H,W) tensors. Returns scalar."""
    loss = 0.0
    weight = 1.0
    p, t = pred.unsqueeze(0), target.unsqueeze(0)
    for i in range(levels):
        # Downsample
        p_down = F.avg_pool2d(p, 2)
        t_down = F.avg_pool2d(t, 2)
        # Upsample back
        p_up = F.interpolate(p_down, size=p.shape[2:], mode='bilinear', align_corners=False)
        t_up = F.interpolate(t_down, size=t.shape[2:], mode='bilinear', align_corners=False)
        # Laplacian = detail at this scale
        p_lap = p - p_up
        t_lap = t - t_up
        loss += weight * torch.abs(p_lap - t_lap).mean()
        # Next scale
        p, t = p_down, t_down
        weight *= 0.5
    # Coarsest residual
    loss += weight * torch.abs(p - t).mean()
    return loss

def sobel_edges(x):
    """Compute Sobel edge magnitude. Input: (1,H,W) or (C,H,W). Output: same shape."""
    needs_batch = (x.dim() == 3)
    if needs_batch:
        x = x.unsqueeze(0)
    c = x.shape[1]
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3).expand(c,-1,-1,-1)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3).expand(c,-1,-1,-1)
    gx = F.conv2d(x, sobel_x, padding=1, groups=c)
    gy = F.conv2d(x, sobel_y, padding=1, groups=c)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    if needs_batch:
        mag = mag.squeeze(0)
    return mag

def texture_loss(pred_rk, target_low, kernel_size=15):
    """Texture loss in LCN space + normalized gradient matching.
    Inputs: (C,H,W) tensors. Returns scalar."""
    # Luminance
    y_pred = rgb_to_y(pred_rk)
    y_target = rgb_to_y(target_low)
    # LCN
    lcn_pred = local_contrast_normalize(y_pred, kernel_size)
    lcn_target = local_contrast_normalize(y_target, kernel_size)
    # Laplacian pyramid on LCN
    lap_loss = laplacian_pyramid_loss(lcn_pred, lcn_target, levels=3)
    # Normalized gradient matching
    grad_pred = sobel_edges(y_pred)
    grad_target = sobel_edges(y_target)
    grad_pred_n = grad_pred / (grad_pred.mean() + 1e-5)
    grad_target_n = grad_target / (grad_target.mean() + 1e-5)
    grad_loss = torch.abs(grad_pred_n - grad_target_n).mean()
    return lap_loss + 0.5 * grad_loss

def edge_aware_denoise_reg(r0, rk, edge_map, lambda_edge=1.0, lambda_flat_tv=0.5, lambda_flat_l1=0.25):
    """Edge-aware denoiser regularization on noise residual.
    r0, rk: (C,H,W). edge_map: (1,H,W) normalized [0,1]."""
    residual = r0 - rk  # noise residual
    flat_map = 1.0 - edge_map
    # Penalize subtracting on edges
    edge_loss = lambda_edge * (edge_map * residual.abs()).mean()
    # TV on flat-region residual
    flat_res = (flat_map * residual).unsqueeze(0)
    h_diff = (flat_res[:,:,1:,:] - flat_res[:,:,:-1,:]).pow(2).mean()
    w_diff = (flat_res[:,:,:,1:] - flat_res[:,:,:,:-1]).pow(2).mean()
    flat_tv = lambda_flat_tv * (h_diff + w_diff)
    # Small L1 on flat residual
    flat_l1 = lambda_flat_l1 * (flat_map * residual.abs()).mean()
    return edge_loss + flat_tv + flat_l1

## ============ Normal Vector Losses ============

def normal_from_depth_bilateral(view, depth, guide_image, sigma_s=5.0, sigma_r=0.1):
    """Compute normals from depth with bilateral filtering guided by structure prior.
    Args:
        view: Camera object (for depth_to_normal)
        depth: (1, H, W) rendered depth
        guide_image: (1, H, W) or (C, H, W) structure prior / edge guide
    Returns:
        normal_map: (3, H, W) normalized surface normals
    """
    from utils.depth_utils import depth_to_normal
    # Bilateral filter on depth guided by structure prior
    d_np = depth[0].detach().cpu().numpy()
    g_np = guide_image[0].detach().cpu().numpy() if guide_image.shape[0] == 1 else guide_image.mean(0).detach().cpu().numpy()
    import cv2
    # Normalize depth to [0,1] for bilateral filter
    d_min, d_max = d_np.min(), d_np.max()
    if d_max - d_min > 1e-6:
        d_norm = ((d_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d_np, dtype=np.uint8)
    d_filtered = cv2.bilateralFilter(d_norm, int(sigma_s * 2 + 1), sigma_r * 255, sigma_s).astype(np.float32)
    d_filtered = torch.tensor(d_filtered / 255.0 * (d_max - d_min) + d_min, device=depth.device)
    # depth_to_normal expects (1, H, W)
    normal_map, _ = depth_to_normal(view, d_filtered.unsqueeze(0))
    # normal_map is (H, W, 3) -> (3, H, W)
    return normal_map.permute(2, 0, 1)


def normal_tv_loss(view, depth, guide_image):
    """Pilot B: Edge-aware TV on bilateral-filtered depth normals.
    Args:
        view: Camera object
        depth: (1, H, W) rendered depth
        guide_image: (1, H, W) structure prior for edge guidance
    Returns:
        scalar loss
    """
    normal_map = normal_from_depth_bilateral(view, depth, guide_image)  # (3, H, W)
    # Edge map from guide for edge-aware weighting
    edge = sobel_edges(guide_image)  # (1, H, W)
    edge = edge / (edge.max() + 1e-5)
    flat_weight = 1.0 - edge  # (1, H, W)
    # TV on normal map, weighted by flat regions
    n = normal_map.unsqueeze(0)  # (1, 3, H, W)
    w = flat_weight.unsqueeze(0)  # (1, 1, H, W)
    h_diff = ((n[:, :, 1:, :] - n[:, :, :-1, :]) * w[:, :, 1:, :]).pow(2).mean()
    w_diff = ((n[:, :, :, 1:] - n[:, :, :, :-1]) * w[:, :, :, 1:]).pow(2).mean()
    return h_diff + w_diff


def normal_axis_loss(normals, rotations, scales):
    """Pilot A: Penalize deviation of learnable normal from shortest-axis direction.
    Args:
        normals: (N, 3) unit normals per Gaussian
        rotations: (N, 4) quaternions
        scales: (N, 3) log-scales (pre-activation)
    Returns:
        scalar loss
    """
    from utils.general_utils import build_rotation
    R = build_rotation(rotations)  # (N, 3, 3)
    # Shortest axis = column of R corresponding to min scale
    scales_act = torch.exp(scales)  # (N, 3)
    min_idx = scales_act.argmin(dim=1)  # (N,)
    # Gather shortest axis column from R
    # R[:, :, i] for each Gaussian
    shortest_axis = R[torch.arange(R.shape[0], device=R.device), :, min_idx]  # (N, 3)
    # Loss: 1 - |dot(n, shortest_axis)|
    dot = (normals * shortest_axis).sum(dim=1).abs()
    return (1.0 - dot).mean()


def normal_consistency_loss(rendered_normal, pseudo_normal, confidence=None):
    """Pilot A: L1 loss between rendered and pseudo-GT normals with optional confidence.
    Args:
        rendered_normal: (3, H, W)
        pseudo_normal: (3, H, W)
        confidence: (1, H, W) optional weight map [0,1]
    Returns:
        scalar loss
    """
    diff = (rendered_normal - pseudo_normal).abs()
    if confidence is not None:
        diff = diff * confidence
    return diff.mean()


def stage_agreement_confidence(denoised_list, view, depth):
    """Pilot C: Compute confidence from denoiser stage agreement on depth-derived normals.
    Higher agreement across stages = higher confidence in that region.
    Args:
        denoised_list: list of (3, H, W) denoised images at each stage
        view: Camera object
        depth: (1, H, W) rendered depth
    Returns:
        confidence: (1, H, W) in [0, 1]
    """
    from utils.depth_utils import depth_to_normal
    if len(denoised_list) < 2:
        return torch.ones(1, depth.shape[1], depth.shape[2], device=depth.device)
    # Use luminance variance across stages as disagreement signal
    luminances = []
    for rk in denoised_list:
        Y = 0.299 * rk[0] + 0.587 * rk[1] + 0.114 * rk[2]
        luminances.append(Y)
    stack = torch.stack(luminances, dim=0)  # (K, H, W)
    variance = stack.var(dim=0)  # (H, W)
    # Normalize: low variance = high confidence
    max_var = variance.max() + 1e-6
    confidence = 1.0 - (variance / max_var)
    # Smooth with small Gaussian blur
    confidence = confidence.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    confidence = torch.nn.functional.avg_pool2d(confidence, kernel_size=5, stride=1, padding=2)
    return confidence.squeeze(0)  # (1, H, W)


def pseudo_gt_luminance_preserve(low_image, exposure=0.37):
    """Luminance-preserving pseudo-GT: scale luminance, preserve chroma.
    Input: (C,H,W). Output: (C,H,W)."""
    Y = rgb_to_y(low_image)
    Y_mean = Y.mean()
    gain = torch.clamp(torch.tensor(exposure, device=low_image.device) / (Y_mean + 1e-6), 1.0, 6.0)
    Y_target = Y * gain
    # Preserve chroma: scale RGB by luminance ratio
    target = low_image * (Y_target / (Y + 1e-6))
    return torch.clamp(target, 0, 1)


# ============ Wave 3 Experiment Losses ============

def log_l1_loss(pred, target, alpha=1.0):
    """Log-space L1 loss for better dark-region gradient flow."""
    return F.l1_loss(torch.log1p(alpha * pred), torch.log1p(alpha * target))


def rgb_to_ycbcr(img):
    """Convert [3,H,W] RGB to YCbCr. ITU-R BT.601."""
    Y  = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    Cb = 0.564 * (img[2] - Y)
    Cr = 0.713 * (img[0] - Y)
    return Y.unsqueeze(0), torch.stack([Cb, Cr], dim=0)


def haar_wavelet_2d(x):
    """Haar wavelet decomposition. x: [C,H,W] -> (LL, LH, HL, HH) each [C,H//2,W//2]."""
    # Pad to even dimensions
    _, H, W = x.shape
    if H % 2 == 1:
        x = F.pad(x, (0, 0, 0, 1), mode='reflect')
    if W % 2 == 1:
        x = F.pad(x, (0, 1, 0, 0), mode='reflect')
    x0 = x[:, 0::2, 0::2]
    x1 = x[:, 0::2, 1::2]
    x2 = x[:, 1::2, 0::2]
    x3 = x[:, 1::2, 1::2]
    LL = (x0 + x1 + x2 + x3) * 0.25
    LH = (x0 - x1 + x2 - x3) * 0.25
    HL = (x0 + x1 - x2 - x3) * 0.25
    HH = (x0 - x1 - x2 + x3) * 0.25
    return LL, LH, HL, HH


def wavelet_residual_loss(r0, rk, lambda_low=1.0, lambda_high=0.1):
    """Wavelet-domain regularization on denoiser residual (R0 - R_K).
    Penalizes low-freq leakage (structure), allows high-freq (noise)."""
    residual = r0 - rk
    LL, LH, HL, HH = haar_wavelet_2d(residual)
    loss_low = LL.abs().mean()
    loss_high = (LH.abs().mean() + HL.abs().mean() + HH.abs().mean()) / 3.0
    return lambda_low * loss_low + lambda_high * loss_high


# ============ Wavelet Fix: Gated HF Losses ============

def _sobel_xy(y):
    """Sobel gradients for [1,1,H,W] input."""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=y.dtype, device=y.device).view(1,1,3,3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return gx, gy


def structure_tensor_mask(r0, win=5, coh_tau=0.35, edge_tau=0.06, temp=0.1):
    """Detect oriented texture regions via structure tensor coherence.
    Returns [1, H, W] mask: high = texture, low = flat/noise."""
    y = (0.299 * r0[0] + 0.587 * r0[1] + 0.114 * r0[2]).unsqueeze(0).unsqueeze(0)
    gx, gy = _sobel_xy(y)
    pad = win // 2
    Jxx = F.avg_pool2d(gx * gx, win, 1, pad)
    Jyy = F.avg_pool2d(gy * gy, win, 1, pad)
    Jxy = F.avg_pool2d(gx * gy, win, 1, pad)
    coh = torch.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-6)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mask = torch.sigmoid((coh - coh_tau) / temp) * torch.sigmoid((edge - edge_tau) / temp)
    return mask.squeeze(0)  # [1, H, W]


def wavelet_struct_gated_loss(r0, rk, lambda_low=1.0, lambda_high=0.2):
    """Structure-tensor gated wavelet: penalize HF only in flat/noise regions."""
    residual = r0 - rk
    LL, LH, HL, HH = haar_wavelet_2d(residual)
    mask = structure_tensor_mask(r0)  # [1, H, W]
    mask_ds = F.avg_pool2d(mask.unsqueeze(0), 2, 2).squeeze(0)  # match wavelet size
    hf = (LH.abs() + HL.abs() + HH.abs()) / 3.0
    hf_loss = ((1.0 - mask_ds) * hf).mean()
    return lambda_low * LL.abs().mean() + lambda_high * hf_loss


def stage_hf_confidence(r0, denoised_list, beta=2.0):
    """Cross-stage HF stability → confidence. Stable HF = texture, unstable = noise."""
    hf_list = []
    for rk_i in denoised_list:
        _, LH, HL, HH = haar_wavelet_2d(r0 - rk_i)
        hf_list.append((LH.abs() + HL.abs() + HH.abs()) / 3.0)
    hf = torch.stack(hf_list, dim=0)  # [K, C, h, w]
    mean = hf.mean(0)
    var = hf.var(0)
    return mean / (mean + beta * var + 1e-6)


def wavelet_stage_gated_loss(r0, denoised_list, lambda_low=1.0, lambda_high=0.2, beta=2.0):
    """Stage-consistency gated wavelet: penalize HF that is unstable across denoiser stages."""
    rk = denoised_list[-1]
    residual = r0 - rk
    LL, LH, HL, HH = haar_wavelet_2d(residual)
    conf = stage_hf_confidence(r0, denoised_list, beta)
    hf = (LH.abs() + HL.abs() + HH.abs()) / 3.0
    hf_loss = ((1.0 - conf) * hf).mean()
    return lambda_low * LL.abs().mean() + lambda_high * hf_loss


def wavelet_ycbcr_loss(r0, rk, lambda_y=1.0, lambda_c=3.0, edge_tau=0.05):
    """Luma-protect / chroma-suppress wavelet: strong HF penalty on chroma, edge-gated on luma."""
    residual = r0 - rk
    y_res = (0.299 * residual[0] + 0.587 * residual[1] + 0.114 * residual[2]).unsqueeze(0)
    cb_res = 0.564 * (residual[2] - y_res.squeeze(0))
    cr_res = 0.713 * (residual[0] - y_res.squeeze(0))
    c_res = torch.stack([cb_res, cr_res], dim=0)
    # Luma HF: edge-gated
    _, y1, y2, y3 = haar_wavelet_2d(y_res)
    edge = sobel_edges(rgb_to_y(r0))  # [1, H, W]
    edge_ds = F.avg_pool2d(edge.unsqueeze(0), 2, 2).squeeze(0)  # [1, H//2, W//2]
    edge_norm = edge_ds / (edge_ds.max() + 1e-5)
    keep = torch.sigmoid((edge_norm - edge_tau) / 0.1)
    loss_y = ((1.0 - keep) * (y1.abs() + y2.abs() + y3.abs()) / 3.0).mean()
    # Chroma HF: full penalty
    _, c1, c2, c3 = haar_wavelet_2d(c_res)
    loss_c = (c1.abs() + c2.abs() + c3.abs()).mean() / 3.0
    # LL penalty on full residual
    LL, _, _, _ = haar_wavelet_2d(residual)
    loss_ll = LL.abs().mean()
    return loss_ll + lambda_y * loss_y + lambda_c * loss_c
