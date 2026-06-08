import os
import math
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, pearson_depth_loss, local_pearson_loss, l2_loss, SmoothLoss, TVLoss, texture_loss, edge_aware_denoise_reg, sobel_edges, rgb_to_y, pseudo_gt_luminance_preserve, normal_tv_loss, normal_axis_loss, normal_consistency_loss, normal_from_depth_bilateral, stage_agreement_confidence, log_l1_loss, rgb_to_ycbcr, wavelet_residual_loss, wavelet_struct_gated_loss, wavelet_stage_gated_loss, wavelet_ycbcr_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from os import makedirs    
from scene.cameras import Camera
import random
import cv2
import numpy as np
import torchvision
from ciconv2d import CIConv2d
from utils.cross_view_loss import cross_view_prior_loss, cross_view_mutex_loss, select_neighbor

# ─── Fitted curve registry (normalized t=x/p90 space) ───
def apply_fitted_curve(t, variant):
    """Apply a fitted tone curve in normalized space. t = low_image / p90."""
    _curves = {
        'poly3_free':     lambda t: 0.0720*t**3 - 0.4376*t**2 + 0.9948*t + 0.1032,
        'poly3_origin':   lambda t: 0.3515*t**3 - 1.1996*t**2 + 1.5639*t,
        'poly5_origin':   lambda t: 0.2331*t**5 - 1.2710*t**4 + 2.6300*t**3 - 2.7450*t**2 + 1.8784*t,
        'aces_filmic':    lambda t: (4.1542*t**2 + 0.1109*t) / (3.1762*t**2 + 2.7073*t + 0.0001),
        'hybrid':         lambda t: 0.6643*(1-torch.exp(-1.4166*t))**0.9383 + 0.2156*torch.clamp(t, 1e-10)**0.4584,
        'softexp_gamma':  lambda t: 1.0348*(1-torch.exp(-0.9861*t))**0.7543,
        'exp_plateau':    lambda t: 1.0799*(1-torch.exp(-1.1150*torch.clamp(t, 1e-10)**0.8076)),
        'rational':       lambda t: (3.4053*t + 50.0*t**2) / (1 + 34.8493*t + 37.8681*t**2 + 1e-10),
        'power_softexp':  lambda t: (1-torch.exp(-1.0916*t))**0.7771,
        'log_curve':      lambda t: 0.3812*torch.log1p(5.5936*t),
        'sinh_ratio':     lambda t: 1.0067*torch.sinh(1.3673*t) / (torch.sinh(1.3673*t) + 0.7115),
        'hable':          lambda t: (0.0001*t**2 + 0.3171*t) / (1.2181*t**2 + 0.2189*t + 0.0001 + 1e-10) * 0.9797,
        'reinhard':       lambda t: 1.2180*t / (t + 0.6763 + 1e-10),
        'softexp_mod':    lambda t: 0.5075*(1-torch.exp(-3.1321*t)) + 0.2293*t,
        'piecewise3':     lambda t: torch.where(t < 0.2513, 0.5473*t,
                            torch.where(t < 0.8845, 0.5473*0.2513 + 0.2111*(t-0.2513),
                              0.5473*0.2513 + 0.2111*(0.8845-0.2513) + 1.4447*(t-0.8845))),
        'bezier3':        lambda t: torch.clamp(t, 0, 1).pow(1) * (3*(1-torch.clamp(t,0,1))**2*0.5561*torch.clamp(t,0,1) + 3*(1-torch.clamp(t,0,1))*torch.clamp(t,0,1)**2*0.9430 + torch.clamp(t,0,1)**3) + (1-torch.clamp(t,0,1).pow(0))*t*0.5637,
        'gamma':          lambda t: 0.7199*torch.clamp(t, 1e-10)**0.5617,
        'double_exp':     lambda t: 0.9233*(1-torch.exp(-1.6887*t)) - 0.0076*(1-torch.exp(-1.6900*t)),
        'sqrt_blend':     lambda t: 0.6186*torch.sqrt(torch.clamp(t, 1e-10)) + 0.0970*t,
        'arctan_curve':   lambda t: 0.6527*torch.atan(2.1182*t) / (torch.pi/2),
        'richards':       lambda t: 5.0*(1+0.1*torch.exp(-1.9246*t))**(-1/0.4813),
        'tanh_curve':     lambda t: 0.8473*torch.tanh(1.4749*t),
    }
    fn = _curves.get(variant)
    if fn is None:
        raise ValueError(f"Unknown curve_variant: {variant}. Available: {list(_curves.keys())}")
    return torch.clamp(fn(t), 0, 1)

def gray(t): return (
        np.clip((t[0][0] if len(t.shape) == 4 else t[0]).detach().cpu().numpy(), 0, 1) * 255).astype(
    np.uint8)

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    return state_dict
    
def prior_loss_cal(model, rendered_prior, target_image, a):

    with torch.no_grad():
        target_features = model(target_image.unsqueeze(0))
    
    rendered_W = torch.clamp(rendered_prior, 0, 1)
    target_W = torch.clamp(target_features, 0, 1).squeeze(0)
    WW = l1_loss(target_W, rendered_W)
    loss = WW * a
    return loss


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Lock cuDNN's convolution autotuner (denoiser CNN). Note: the diff
        # Gaussian rasterizer backward uses atomicAdd and has no deterministic
        # implementation, so this does NOT guarantee bit-exact reproducibility —
        # it only removes cuDNN as a nondeterminism source. Whether that is
        # enough to make same-seed runs reproduce is exactly what we test.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder 
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training(dataset, save_dir, opt, pipe, testing_iterations, saving_iterations, debug_from):
    #loss_smooth = SmoothLoss().cuda()
    tv_loss = TVLoss()
    
    prior_model = CIConv2d('W', k=3, scale=0.8)
    prior_model = prior_model.cuda()

    set_seed(opt.training_seed, getattr(opt, 'deterministic', False))
    first_iter = 0
    dataset.model_path = save_dir
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(sh_degree=dataset.sh_degree, depth_feat_dim=1, prior_feat_dim=1, noise_feat_dim=3, illu_feat_dim=3, denoiser_stage=dataset.denoiser_stage, denoiser_layers=dataset.denoiser_layers, denoiser_channel=dataset.denoiser_channel, denoiser_mode=dataset.denoiser_mode) 
    scene = Scene(dataset, gaussians, opt.use_depth, opt.use_prior, gap = pipe.interval) 

    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    trainCameras = scene.getTrainCameras().copy()
    all_train_cams = trainCameras
    gaussians.compute_3D_filter(cameras=trainCameras)

    # Preload precomputed pseudo-GT images if mode is "precomputed"
    precomputed_gt = {}
    if getattr(opt, 'pseudo_gt_mode', '') == "precomputed":
        _pg_dir = getattr(opt, 'precomputed_gt_dir', '')
        if _pg_dir and os.path.isdir(_pg_dir):
            from torchvision import transforms as _T
            _to_tensor = _T.ToTensor()
            from PIL import Image as _PILImage
            for _cam in trainCameras:
                _name = _cam.image_name
                for _ext in ['.JPG', '.jpg', '.png', '.PNG']:
                    _p = os.path.join(_pg_dir, _name + _ext)
                    if os.path.exists(_p):
                        _img = _to_tensor(_PILImage.open(_p).convert('RGB'))
                        # Resize to match camera resolution
                        if _img.shape[1] != _cam.image_height or _img.shape[2] != _cam.image_width:
                            _img = F.interpolate(_img.unsqueeze(0), size=(_cam.image_height, _cam.image_width), mode='bilinear', align_corners=False).squeeze(0)
                        precomputed_gt[_name] = _img.clamp(0, 1).cuda()
                        break
            print(f"[precomputed] Loaded {len(precomputed_gt)} pseudo-GT images from {_pg_dir}")
        else:
            print(f"[precomputed] WARNING: dir not found: {_pg_dir}")

    # Precompute scene-level luminance statistics for adaptive_softexp_v3_scene.
    # ASE-v2 recomputes p90/p10/mean_Y per image, so the softexp curve SHAPE
    # (g enters the exponent) varies view-to-view, which destabilises training
    # under different seeds. v3_scene fixes one curve per scene by averaging
    # per-view percentile stats, matching AP3's fixed-shape behaviour.
    scene_stats = {}
    if getattr(opt, 'pseudo_gt_mode', '') == "adaptive_softexp_v3_scene":
        _p90s, _p10s, _meanYs = [], [], []
        for _cam in trainCameras:
            _li = _cam.low_image
            _Y = (0.299 * _li[0] + 0.587 * _li[1] + 0.114 * _li[2]).flatten()
            _p90s.append(torch.quantile(_Y, 0.90).item())
            _p10s.append(torch.quantile(_Y, 0.10).item())
            _meanYs.append(_Y.mean().item())
        _n = max(1, len(_p90s))
        scene_stats['p90'] = sum(_p90s) / _n + 1e-6
        scene_stats['p10'] = sum(_p10s) / _n + 1e-6
        scene_stats['mean_Y'] = sum(_meanYs) / _n
        print(f"[ase_v3_scene] scene-level p90={scene_stats['p90']:.4f} "
              f"p10={scene_stats['p10']:.4f} mean_Y={scene_stats['mean_Y']:.4f} "
              f"over {len(trainCameras)} train views")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_l1_loss_for_log, ema_ssim_loss_for_log =0.0, 0.0

    best_psnr = 0
    best_ssim = 0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        randindex = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam: Camera = viewpoint_stack.pop(randindex)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, depth_threshold = opt.depth_threshold * scene.cameras_extent, iteration = iteration, achromatic_illu = opt.achromatic_illu)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"])
        rendered_depth: torch.Tensor = render_pkg["depth"]
        rendered_prior: torch.Tensor = render_pkg["prior"]

        rendered_low: torch.Tensor = render_pkg["render_low"]
        rendered_denoised_list: torch.Tensor = render_pkg["denoised_list"]

        target = opt.exposure
        low_image = viewpoint_cam.low_image
        low_mean = low_image.mean()
        if opt.use_gt_as_pseudo:
            target_image = viewpoint_cam.original_image.clone()
        elif opt.pseudo_gt_mode == "p80_softexp":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            q80 = torch.quantile(Y.flatten(), 0.80)
            gain = torch.clamp(target / (q80 + 1e-6), 1.0, 6.0)
            target_image = 1.0 - torch.exp(-gain * low_image)
        elif opt.pseudo_gt_mode == "lum_preserve":
            target_image = pseudo_gt_luminance_preserve(low_image, target)
        elif opt.pseudo_gt_mode == "per_channel":
            target_image = torch.zeros_like(low_image)
            for c in range(3):
                ch_mean = low_image[c].mean() + 1e-6
                target_image[c] = torch.clamp(low_image[c] * (target / ch_mean), 0, 1)
        elif opt.pseudo_gt_mode == "gt_color_transfer":
            # Log-chroma moment matching with confidence mask
            target_image = torch.clamp(low_image * (target / low_mean), 0, 1)
            eps = 1e-6
            g_ch = target_image[1] + eps
            cur_log_bg = torch.log(target_image[2] / g_ch + eps).mean()
            cur_log_rg = torch.log(target_image[0] / g_ch + eps).mean()
            # Confidence mask: only correct in bright-enough pixels
            Y = 0.299 * target_image[0] + 0.587 * target_image[1] + 0.114 * target_image[2]
            conf = torch.clamp(Y / 0.1, 0, 1)
            alpha = opt.color_transfer_alpha
            bg_shift = alpha * (opt.gt_log_bg_mean - cur_log_bg)
            rg_shift = alpha * (opt.gt_log_rg_mean - cur_log_rg)
            corrected = target_image.clone()
            corrected[2] = target_image[2] * torch.exp(bg_shift)
            corrected[0] = target_image[0] * torch.exp(rg_shift)
            target_image = conf * corrected + (1 - conf) * target_image
            target_image = torch.clamp(target_image, 0, 1)
        elif opt.pseudo_gt_mode == "blue_boost":
            target_image = torch.clamp(low_image * (target / low_mean), 0, 1)
            target_image[2] = torch.clamp(target_image[2] * opt.blue_boost_factor, 0, 1)
        elif opt.pseudo_gt_mode == "blue_boost_conf":
            target_image = torch.clamp(low_image * (target / low_mean), 0, 1)
            Y = 0.299 * target_image[0] + 0.587 * target_image[1] + 0.114 * target_image[2]
            conf = torch.clamp(Y / 0.1, 0, 1)
            boosted = torch.clamp(target_image[2] * opt.blue_boost_factor, 0, 1)
            target_image[2] = conf * boosted + (1 - conf) * target_image[2]
        elif opt.pseudo_gt_mode == "adaptive_softexp":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            gain = -torch.log(torch.tensor(0.2, device=low_image.device)) / p90
            target_image = 1.0 - torch.exp(-gain * low_image)
            if opt.softexp_power != 1.0:
                target_image = torch.clamp(target_image, 1e-8, 1.0) ** opt.softexp_power
        elif opt.pseudo_gt_mode == "sigmoid":
            from experiments.sigmoid_search.sigmoid_functions import apply_sigmoid
            target_image = apply_sigmoid(low_image, opt.sigmoid_variant, opt.exposure)
        elif opt.pseudo_gt_mode == "fitted_curve":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            t = low_image / p90
            target_image = apply_fitted_curve(t, opt.curve_variant)
        elif opt.pseudo_gt_mode == "adaptive_sigmoid":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            k = opt.sigmoid_k_base
            t = low_image / p90
            y_core = torch.tanh(k * t)
            target_image = torch.clamp(offset + (1.0 - offset) * y_core, 0, 1)
        elif opt.pseudo_gt_mode == "adaptive_poly3":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            mean_Y = Y.mean().item()
            brightness_gain = 1.0 + opt.g_adapt * mean_Y
            t = low_image / p90
            poly = 0.0720*t**3 - 0.4376*t**2 + 0.9948*t
            target_image = torch.clamp(offset + poly * brightness_gain, 0, 1)
        elif opt.pseudo_gt_mode == "adaptive_softexp_v2":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            mean_Y = Y.mean().item()
            g = opt.sigmoid_k_base + opt.g_adapt * mean_Y
            t = low_image / p90
            target_image = offset + (1.0 - offset) * (1.0 - torch.exp(-g * t))
            target_image = torch.clamp(target_image, 0, 1)
        elif opt.pseudo_gt_mode == "adaptive_softexp_v3_scene":
            # Scene-level ASE: one fixed curve shape for all views (stats
            # precomputed once before the training loop). Same formula as
            # adaptive_softexp_v2 but p90/p10/mean_Y are scene constants.
            p90 = scene_stats['p90']
            p10 = scene_stats['p10']
            r = max(0.0, min(1.0, p10 / p90))
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            g = opt.sigmoid_k_base + opt.g_adapt * scene_stats['mean_Y']
            t = low_image / p90
            target_image = offset + (1.0 - offset) * (1.0 - torch.exp(-g * t))
            target_image = torch.clamp(target_image, 0, 1)
        elif opt.pseudo_gt_mode == "adaptive_softexp_v4_floor":
            # Stable-ASE: ASE-v2 with a linear bypass that gives T'(t) >= A*lam > 0.
            # T_S(t) = offset + A * [(1-lam)*(1-exp(-g t)) + lam*t]
            # with A = (1-offset) * z1 / [(1-lam)*z1 + lam],  z1 = 1 - exp(-g)
            # so T_S(1) matches v2's value at t=1. lam=0 reproduces v2.
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            mean_Y = Y.mean().item()
            g = opt.sigmoid_k_base + opt.g_adapt * mean_Y
            lam = opt.softexp_linear_lambda
            t = low_image / p90
            z = 1.0 - torch.exp(-g * t)
            z1 = 1.0 - math.exp(-g)
            A = (1.0 - offset) * z1 / ((1.0 - lam) * z1 + lam + 1e-8)
            target_image = offset + A * ((1.0 - lam) * z + lam * t)
            target_image = torch.clamp(target_image, 0, 1)
        elif opt.pseudo_gt_mode == "precomputed":
            _name = viewpoint_cam.image_name
            if _name in precomputed_gt:
                target_image = precomputed_gt[_name]
            else:
                target_image = torch.clamp(low_image * (target / low_mean), 0, 1)
        else:
            target_image = torch.clamp(low_image * (target/low_mean), 0, 1)

        loss = 0

        if opt.loss_mode == "3branch" and opt.use_denoiser:
            # ============ 3-Branch Loss ============
            rk = rendered_denoised_list[-1]

            # Branch 1: Color — L_rgb on R0 (direct gradients to Gaussians)
            loss_l1 = l1_loss(rendered_image, target_image)
            loss_ssim = (1.0 - ssim(rendered_image, target_image.unsqueeze(0)))
            rgb_loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * loss_ssim
            loss += rgb_loss * opt.lambda_rgb

            # Branch 2: Texture — LCN + Laplacian pyramid on R_K (after warmup)
            if iteration >= opt.texture_warmup_iter:
                loss_tex = texture_loss(rk, low_image)
                loss += opt.lambda_texture * loss_tex

            # Branch 3: Structure — depth + prior (unchanged)
            if opt.use_depth:
                gt_depth = viewpoint_cam.gt_depth.unsqueeze(0).to(dataset.data_device)
                pearson_loss_val = pearson_depth_loss(rendered_depth[0], gt_depth[0])
                lp_loss = local_pearson_loss(rendered_depth[0], gt_depth[0], 128, 0.5)
                depth_loss = (pearson_loss_val + lp_loss) * opt.lambda_depth
                loss += depth_loss

            if opt.use_prior:
                prior_loss = prior_loss_cal(prior_model, rendered_prior, target_image, opt.lambda_prior)
                loss += prior_loss

            # Cross-view prior consistency & mutex
            _need_nb = (opt.lambda_xv > 0 and iteration >= opt.xv_warmup_iter) or \
                       (opt.lambda_mutex > 0 and iteration >= opt.mutex_warmup_iter)
            if _need_nb:
                nb_cam = select_neighbor(viewpoint_cam, all_train_cams, opt.xv_num_neighbors)
                with torch.no_grad():
                    nb_pkg = render(nb_cam, gaussians, pipe, background,
                                   depth_threshold=opt.depth_threshold * scene.cameras_extent,
                                   iteration=iteration)
                nb_prior = nb_pkg["prior"]
                nb_depth = nb_pkg["depth"]
                if opt.lambda_xv > 0 and iteration >= opt.xv_warmup_iter:
                    xv_loss = cross_view_prior_loss(rendered_prior, rendered_depth, nb_prior, nb_depth, viewpoint_cam, nb_cam)
                    loss += opt.lambda_xv * xv_loss
                if opt.lambda_mutex > 0 and iteration >= opt.mutex_warmup_iter:
                    mutex_loss = cross_view_mutex_loss(
                        rendered_prior, rendered_depth, nb_prior, nb_depth,
                        render_pkg["illu"], render_pkg["noise"],
                        viewpoint_cam, nb_cam, alpha=opt.mutex_noise_alpha)
                    loss += opt.lambda_mutex * mutex_loss
            with torch.no_grad():
                edge_map = sobel_edges(rgb_to_y(low_image))
                edge_map = edge_map / (edge_map.max() + 1e-5)
            loss_denoise = edge_aware_denoise_reg(rendered_image, rk, edge_map)
            loss += opt.lambda_denoise_reg * loss_denoise

            # ============ Normal Vector Losses ============
            if iteration >= opt.normal_warmup_iter:
                # Pilot B: Edge-aware TV on depth-derived normals
                if opt.lambda_normal_tv > 0:
                    loss_ntv = normal_tv_loss(viewpoint_cam, rendered_depth, rendered_prior)
                    loss += opt.lambda_normal_tv * loss_ntv

                # Pilot A: Learnable normal losses
                if opt.use_learnable_normals and gaussians.get_normals is not None:
                    # Axis alignment loss (always active when normals enabled)
                    if opt.lambda_normal_axis > 0:
                        loss_naxis = normal_axis_loss(gaussians.get_normals, gaussians._rotation, gaussians._scaling)
                        loss += opt.lambda_normal_axis * loss_naxis

                    # Consistency with depth-derived pseudo-GT normals
                    if opt.lambda_normal_consistency > 0:
                        with torch.no_grad():
                            pseudo_normal = normal_from_depth_bilateral(viewpoint_cam, rendered_depth, rendered_prior)
                            # Pilot C: Stage-agreement confidence weighting
                            if opt.use_stage_confidence and len(rendered_denoised_list) >= 2:
                                confidence = stage_agreement_confidence(rendered_denoised_list, viewpoint_cam, rendered_depth)
                            else:
                                confidence = None
                        # Render normals via alpha-blending (use depth channel trick: dot product proxy)
                        # Since we can't easily add channels to rasterizer, use per-Gaussian loss directly
                        loss_ncons = normal_consistency_loss(
                            normal_from_depth_bilateral(viewpoint_cam, rendered_depth, rendered_prior),
                            pseudo_normal, confidence)
                        loss += opt.lambda_normal_consistency * loss_ncons

            # Low-light branch: detach R_K if configured
            if opt.detach_rk_low:
                rendered_low_input = render_pkg["render_low_detached"] if "render_low_detached" in render_pkg else rendered_low
                # Fallback: recompute with detached R_K
                if "render_low_detached" not in render_pkg:
                    # Use rendered_low but stop gradient from R_K
                    rendered_low_input = rendered_low.detach()
                    # Re-forward tonemapper with detached R_K — approximate by detaching rendered_low
            else:
                rendered_low_input = rendered_low

            if iteration >= opt.low_warmup_iter:
                loss_l1_low = l1_loss(rendered_low_input, low_image)
                loss_ssim_low = (1.0 - ssim(rendered_low_input, low_image.unsqueeze(0)))
                rgb_loss_low = (1.0 - opt.lambda_dssim_low) * loss_l1_low + opt.lambda_dssim_low * loss_ssim_low
                loss += rgb_loss_low

        else:
            # ============ Original Loss (backward compatible) ============
            # RGB supervision target: R_K (denoised) or R0 (raw render)
            if opt.rgb_supervision_target == "rk" and opt.use_denoiser:
                rgb_target = rendered_denoised_list[-1]
            else:
                rgb_target = rendered_image
            loss_l1 = l1_loss(rgb_target, target_image)
            loss_ssim = (1.0 - ssim(rgb_target, target_image.unsqueeze(0)))
            rgb_loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * loss_ssim
            loss += rgb_loss * opt.lambda_rgb

            # --- Wave 3: Log-L1 (default OFF) ---
            if opt.lambda_log_rgb > 0:
                loss += opt.lambda_log_rgb * log_l1_loss(rgb_target, target_image, opt.log_rgb_alpha)

            # --- Wave 3: YCbCr luma-first (default OFF, replaces L1 above when ON) ---
            if opt.use_ycbcr_loss:
                pred_Y, pred_C = rgb_to_ycbcr(rgb_target)
                tgt_Y, tgt_C = rgb_to_ycbcr(target_image)
                loss_y = F.l1_loss(pred_Y, tgt_Y)
                loss_cbcr = F.l1_loss(pred_C, tgt_C)
                # Replace the L1 portion: subtract original L1, add YCbCr version
                loss = loss - (1.0 - opt.lambda_dssim) * loss_l1 * opt.lambda_rgb
                loss += (loss_y + opt.lambda_chroma * loss_cbcr) * opt.lambda_rgb

            # Optional R0 auxiliary loss (warm-start)
            if opt.rgb_supervision_target == "rk" and opt.use_denoiser and opt.lambda_rgb_r0_aux > 0 and iteration <= opt.rgb_aux_until_iter:
                loss_l1_r0 = l1_loss(rendered_image, target_image)
                loss_ssim_r0 = (1.0 - ssim(rendered_image, target_image.unsqueeze(0)))
                rgb_loss_r0 = (1.0 - opt.lambda_dssim) * loss_l1_r0 + opt.lambda_dssim * loss_ssim_r0
                loss += opt.lambda_rgb_r0_aux * rgb_loss_r0

            if opt.use_depth:
                gt_depth = viewpoint_cam.gt_depth.unsqueeze(0).to(dataset.data_device)
                pearson_loss = pearson_depth_loss(rendered_depth[0], gt_depth[0])
                lp_loss = local_pearson_loss(rendered_depth[0], gt_depth[0], 128, 0.5)
                depth_loss = (pearson_loss + lp_loss) * opt.lambda_depth
                loss += depth_loss

            if opt.use_prior:
                prior_loss = prior_loss_cal(prior_model, rendered_prior, target_image, opt.lambda_prior)
                loss += prior_loss

            # Cross-view prior consistency & mutex
            _need_nb = (opt.lambda_xv > 0 and iteration >= opt.xv_warmup_iter) or \
                       (opt.lambda_mutex > 0 and iteration >= opt.mutex_warmup_iter)
            if _need_nb:
                nb_cam = select_neighbor(viewpoint_cam, all_train_cams, opt.xv_num_neighbors)
                with torch.no_grad():
                    nb_pkg = render(nb_cam, gaussians, pipe, background,
                                   depth_threshold=opt.depth_threshold * scene.cameras_extent,
                                   iteration=iteration)
                nb_prior = nb_pkg["prior"]
                nb_depth = nb_pkg["depth"]
                if opt.lambda_xv > 0 and iteration >= opt.xv_warmup_iter:
                    xv_loss = cross_view_prior_loss(rendered_prior, rendered_depth, nb_prior, nb_depth, viewpoint_cam, nb_cam)
                    loss += opt.lambda_xv * xv_loss
                if opt.lambda_mutex > 0 and iteration >= opt.mutex_warmup_iter:
                    mutex_loss = cross_view_mutex_loss(
                        rendered_prior, rendered_depth, nb_prior, nb_depth,
                        render_pkg["illu"], render_pkg["noise"],
                        viewpoint_cam, nb_cam, alpha=opt.mutex_noise_alpha)
                    loss += opt.lambda_mutex * mutex_loss

            if opt.use_denoiser:
                loss_l1_low = l1_loss(rendered_low, low_image)
                loss_ssim_low = (1.0 - ssim(rendered_low, low_image.unsqueeze(0)))
                rgb_loss_low = (1.0 - opt.lambda_dssim_low) * loss_l1_low + opt.lambda_dssim_low * loss_ssim_low
                loss += rgb_loss_low

                l2_loss_denoiser = l2_loss(rendered_denoised_list[-1], rendered_image)
                # TV target: noise residual or clean image
                if opt.tv_target == "residual":
                    tv_input = (rendered_image - rendered_denoised_list[-1]).unsqueeze(0)
                else:
                    tv_input = rendered_denoised_list[-1].unsqueeze(0)
                loss_tv = tv_loss(tv_input)
                loss = loss + opt.lambda_denoiser * l2_loss_denoiser + opt.lambda_tv * loss_tv

                # --- Wave 3: Wavelet residual regularization (default OFF) ---
                if opt.lambda_wavelet > 0:
                    if opt.wavelet_mode == "struct":
                        loss += opt.lambda_wavelet * wavelet_struct_gated_loss(
                            rendered_image, rendered_denoised_list[-1],
                            opt.wavelet_lambda_low, opt.wavelet_lambda_high)
                    elif opt.wavelet_mode == "stage":
                        loss += opt.lambda_wavelet * wavelet_stage_gated_loss(
                            rendered_image, rendered_denoised_list,
                            opt.wavelet_lambda_low, opt.wavelet_lambda_high, opt.wavelet_stage_beta)
                    elif opt.wavelet_mode == "ycbcr":
                        loss += opt.lambda_wavelet * wavelet_ycbcr_loss(
                            rendered_image, rendered_denoised_list[-1],
                            opt.wavelet_lambda_high, opt.wavelet_ycbcr_lambda_c)
                    else:
                        loss += opt.lambda_wavelet * wavelet_residual_loss(
                            rendered_image, rendered_denoised_list[-1],
                            opt.wavelet_lambda_low, opt.wavelet_lambda_high)
                # --- Flat-region TV fallback (default OFF) ---
                if opt.lambda_tv_flat > 0:
                    tv_flat_input = rendered_denoised_list[-1].unsqueeze(0)
                    loss += opt.lambda_tv_flat * tv_loss(tv_flat_input)
     
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_l1_loss_for_log = 0.4 * loss_l1.item() + 0.6 * ema_l1_loss_for_log
            ema_ssim_loss_for_log = 0.4 * loss_ssim.item() + 0.6 * ema_ssim_loss_for_log


            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "L1": f"{ema_l1_loss_for_log:.{4}f}", "SSIM": f"{ema_ssim_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            if iteration in testing_iterations:
                psnr_test, ssim_test = evaluate(opt.use_denoiser, iteration, testing_iterations, scene, render, (pipe, background))
                if psnr_test >= best_psnr or ssim_test >= best_ssim or iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    if psnr_test >= best_psnr:
                        best_psnr = psnr_test
                    if ssim_test >= best_ssim:
                        best_ssim = ssim_test

                print("\n[Best PNSR: {}   Best SSIM: {}]".format(best_psnr, best_ssim))

            elif iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter: 
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)  

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity_3dgs()
                
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    # ---- Save pseudo-GT for all training cameras after training ----
    pseudo_gt_dir = os.path.join(save_dir, "pseudo_gt")
    makedirs(pseudo_gt_dir, exist_ok=True)
    print(f"\n[INFO] Saving pseudo-GT images to {pseudo_gt_dir}")
    for idx, viewpoint_cam in enumerate(all_train_cams):
        low_image = viewpoint_cam.low_image
        low_mean = low_image.mean()
        target = opt.exposure
        if hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "p80_softexp":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            q80 = torch.quantile(Y.flatten(), 0.80)
            gain = torch.clamp(target / (q80 + 1e-6), 1.0, 6.0)
            target_image = 1.0 - torch.exp(-gain * low_image)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "adaptive_softexp":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            gain = -torch.log(torch.tensor(0.2, device=low_image.device)) / p90
            target_image = 1.0 - torch.exp(-gain * low_image)
            if hasattr(opt, 'softexp_power') and opt.softexp_power != 1.0:
                target_image = torch.clamp(target_image, 1e-8, 1.0) ** opt.softexp_power
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "sigmoid":
            from experiments.sigmoid_search.sigmoid_functions import apply_sigmoid
            target_image = apply_sigmoid(low_image, opt.sigmoid_variant, opt.exposure)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "fitted_curve":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            t = low_image / p90
            target_image = apply_fitted_curve(t, opt.curve_variant)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "adaptive_sigmoid":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            k = opt.sigmoid_k_base
            t = low_image / p90
            y_core = torch.tanh(k * t)
            target_image = torch.clamp(offset + (1.0 - offset) * y_core, 0, 1)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "adaptive_poly3":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            mean_Y = Y.mean().item()
            brightness_gain = 1.0 + getattr(opt, 'g_adapt', 0.0) * mean_Y
            t = low_image / p90
            poly = 0.0720*t**3 - 0.4376*t**2 + 0.9948*t
            target_image = torch.clamp(offset + poly * brightness_gain, 0, 1)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "adaptive_softexp_v2":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            mean_Y = Y.mean().item()
            g = opt.sigmoid_k_base + getattr(opt, 'g_adapt', 0.0) * mean_Y
            t = low_image / p90
            target_image = offset + (1.0 - offset) * (1.0 - torch.exp(-g * t))
            target_image = torch.clamp(target_image, 0, 1)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "adaptive_softexp_v3_scene":
            p90 = scene_stats['p90']
            p10 = scene_stats['p10']
            r = max(0.0, min(1.0, p10 / p90))
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            g = opt.sigmoid_k_base + getattr(opt, 'g_adapt', 0.0) * scene_stats['mean_Y']
            t = low_image / p90
            target_image = offset + (1.0 - offset) * (1.0 - torch.exp(-g * t))
            target_image = torch.clamp(target_image, 0, 1)
        elif hasattr(opt, 'pseudo_gt_mode') and opt.pseudo_gt_mode == "adaptive_softexp_v4_floor":
            Y = 0.299 * low_image[0] + 0.587 * low_image[1] + 0.114 * low_image[2]
            p90 = torch.quantile(Y.flatten(), 0.90) + 1e-6
            p10 = torch.quantile(Y.flatten(), 0.10) + 1e-6
            r = (p10 / p90).clamp(0, 1)
            offset = opt.sigmoid_alpha * (1.0 - r) ** opt.sigmoid_beta
            mean_Y = Y.mean().item()
            g = opt.sigmoid_k_base + getattr(opt, 'g_adapt', 0.0) * mean_Y
            lam = getattr(opt, 'softexp_linear_lambda', 0.0)
            t = low_image / p90
            z = 1.0 - torch.exp(-g * t)
            z1 = 1.0 - math.exp(-g)
            A = (1.0 - offset) * z1 / ((1.0 - lam) * z1 + lam + 1e-8)
            target_image = offset + A * ((1.0 - lam) * z + lam * t)
            target_image = torch.clamp(target_image, 0, 1)
        else:
            target_image = torch.clamp(low_image * (target / low_mean), 0, 1)
        torchvision.utils.save_image(target_image, os.path.join(pseudo_gt_dir, '{0:05d}.png'.format(idx)))
    print(f"[INFO] Saved {len(all_train_cams)} pseudo-GT images")

    
def evaluate(use_denoiser, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    psnr_test = 0.0
    ssim_test = 0.0
    if iteration in testing_iterations:
        with open(scene.model_path + "/eval_result.txt", "a") as f:
            f.write(str(iteration) + '  total_points:  ' + str(scene.gaussians.get_xyz.shape[0]) + '\n')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['name'] == 'train':
                continue
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                render_path = os.path.join(scene.model_path, config['name'], "ours_{}".format(iteration), "renders")

                makedirs(render_path, exist_ok=True)

                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    if use_denoiser:
                        image = torch.clamp(render_result["denoised_list"][-1], 0.0, 1.0)
                    else:
                        image = torch.clamp(render_result["render"][-1], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                with open(scene.model_path + "/eval_result.txt", "a") as f:
                    f.write(str(iteration) + '  PSNR:  ' + str(psnr_test) + '  SSIM:  ' + str(ssim_test) + '\n')

        torch.cuda.empty_cache()
        return psnr_test, ssim_test

if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000] + list(range(1500, 7_0001, 100)) + list(range(6000, 30_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 4000, 5000] + list(range(6000, 30_0001, 5000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default =None)
    parser.add_argument("--configs", type=str, default = "")


    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    #safe_state(args.quiet)
    training(dataset=lp.extract(args),
             save_dir=args.model_path,
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
