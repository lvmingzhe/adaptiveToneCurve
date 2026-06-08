import torch
import torch.nn.functional as F
import math
import random


def build_intrinsic(cam):
    """Build 3x3 intrinsic matrix from Camera's FoV and image size."""
    W = cam.image_width
    H = cam.image_height
    fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
    fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
    K = torch.tensor([
        [fx, 0.0, W / 2.0],
        [0.0, fy, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device="cuda")
    return K


def get_w2c(cam):
    """Get world-to-camera [4,4] from Camera (stored as transposed)."""
    return cam.world_view_transform.T


def select_neighbor(cam_a, all_cams, k=5):
    """Select a random neighbor from the k nearest cameras by center distance."""
    center_a = cam_a.camera_center  # [3]
    dists = []
    for c in all_cams:
        if c.uid == cam_a.uid:
            dists.append(float('inf'))
        else:
            dists.append((c.camera_center - center_a).norm().item())
    # Get top-k nearest indices
    topk = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
    return all_cams[random.choice(topk)]


def warp_prior(D_a, P_b, D_b, cam_a, cam_b):
    """
    Warp prior P_b from view b into view a using depth D_a.

    Args:
        D_a: [1, H, W] depth map of view a
        P_b: [1, H, W] prior map of view b
        D_b: [1, H, W] depth map of view b
        cam_a, cam_b: Camera objects

    Returns:
        P_warped: [1, H, W] warped prior from b into a's frame
        mask: [1, H, W] valid pixel mask
    """
    H, W = D_a.shape[1], D_a.shape[2]
    device = D_a.device

    K_a = build_intrinsic(cam_a)
    K_b = build_intrinsic(cam_b)
    W2C_a = get_w2c(cam_a)  # [4,4]
    W2C_b = get_w2c(cam_b)  # [4,4]

    # Pixel grid for view a
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                          torch.arange(W, device=device, dtype=torch.float32),
                          indexing='ij')
    ones = torch.ones_like(u)
    # [3, H*W]
    uv1 = torch.stack([u, v, ones], dim=0).reshape(3, -1)

    # Unproject to camera-a 3D: X_a = D_a * K_a^{-1} * [u,v,1]
    K_a_inv = torch.inverse(K_a)
    depth_flat = D_a.reshape(1, -1)  # [1, H*W]
    pts_cam_a = K_a_inv @ uv1  # [3, H*W]
    pts_cam_a = pts_cam_a * depth_flat  # [3, H*W]

    # Camera a → world: X_w = W2C_a^{-1} * [X_a; 1]
    W2C_a_inv = torch.inverse(W2C_a)
    pts_h = torch.cat([pts_cam_a, ones.reshape(1, -1)], dim=0)  # [4, H*W]
    pts_world = W2C_a_inv @ pts_h  # [4, H*W]

    # World → camera b: X_b = W2C_b * X_w
    pts_cam_b = W2C_b @ pts_world  # [4, H*W]
    z_b = pts_cam_b[2, :]  # [H*W]

    # Project to view b pixels
    pts_proj = K_b @ pts_cam_b[:3, :]  # [3, H*W]
    u_b = pts_proj[0, :] / (z_b + 1e-8)
    v_b = pts_proj[1, :] / (z_b + 1e-8)

    # Normalize to [-1, 1] for grid_sample
    u_norm = 2.0 * u_b / (W - 1) - 1.0
    v_norm = 2.0 * v_b / (H - 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).reshape(1, H, W, 2)

    # Sample P_b and D_b at projected locations
    P_warped = F.grid_sample(P_b.unsqueeze(0), grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True).squeeze(0)  # [1, H, W]
    D_b_sampled = F.grid_sample(D_b.unsqueeze(0), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True).squeeze(0)  # [1, H, W]

    # Visibility mask
    z_b_map = z_b.reshape(1, H, W)
    in_bounds = (u_norm.reshape(1, H, W).abs() <= 1.0) & (v_norm.reshape(1, H, W).abs() <= 1.0)
    z_positive = z_b_map > 0
    depth_consistent = (torch.abs(z_b_map - D_b_sampled) / (z_b_map + 1e-8)) < 0.05
    mask = (in_bounds & z_positive & depth_consistent).float()

    return P_warped, mask


def cross_view_prior_loss(P_a, D_a, P_b, D_b, cam_a, cam_b):
    """
    Cross-view prior consistency loss.

    Args:
        P_a: [1, H, W] prior of anchor view (has gradients)
        D_a: [1, H, W] depth of anchor view (has gradients)
        P_b: [1, H, W] prior of neighbor view (no_grad)
        D_b: [1, H, W] depth of neighbor view (no_grad)
        cam_a, cam_b: Camera objects

    Returns:
        scalar loss
    """
    P_warped, mask = warp_prior(D_a, P_b, D_b, cam_a, cam_b)

    # Minimum coverage check
    coverage = mask.sum() / mask.numel()
    if coverage < 0.05:
        return D_a.sum() * 0.0  # return zero with grad connectivity

    # Affine compensation (detached): alpha * P_warped + beta ≈ P_a
    with torch.no_grad():
        P_a_masked = (P_a * mask).sum() / (mask.sum() + 1e-8)
        P_w_masked = (P_warped * mask).sum() / (mask.sum() + 1e-8)
        cov = ((P_a - P_a_masked) * (P_warped - P_w_masked) * mask).sum() / (mask.sum() + 1e-8)
        var_w = ((P_warped - P_w_masked) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        alpha = cov / (var_w + 1e-8)
        beta = P_a_masked - alpha * P_w_masked

    P_comp = alpha * P_warped + beta

    # Masked L1 + SSIM
    l1 = (torch.abs(P_a - P_comp) * mask).sum() / (mask.sum() + 1e-8)

    # SSIM on masked region (use full images, mask applied after)
    P_a_4d = P_a.unsqueeze(0)       # [1, 1, H, W]
    P_comp_4d = P_comp.unsqueeze(0)  # [1, 1, H, W]
    mask_4d = mask.unsqueeze(0)      # [1, 1, H, W]
    ssim_map = _ssim_map(P_a_4d, P_comp_4d)  # [1, 1, H, W]
    ssim_loss = ((1.0 - ssim_map) * mask_4d).sum() / (mask_4d.sum() + 1e-8)

    loss = 0.8 * l1 + 0.2 * ssim_loss
    return loss


def _sobel_gradient_mag(x):
    """Compute gradient magnitude using Sobel operator. x: [1, H, W] → [1, H, W]."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    x_4d = x.unsqueeze(0)  # [1, 1, H, W]
    gx = F.conv2d(x_4d, sobel_x, padding=1)
    gy = F.conv2d(x_4d, sobel_y, padding=1)
    return (gx ** 2 + gy ** 2).sqrt().squeeze(0)  # [1, H, W]


def cross_view_mutex_loss(P_a, D_a, P_b, D_b, illu_a, noise_a, cam_a, cam_b, alpha=0.5):
    """
    Cross-view structure-degradation mutual exclusion loss.

    Penalizes correlation between warped neighbor structure gradient and
    anchor degradation gradient. Gradients flow only through illu_a/noise_a.

    Args:
        P_a: [1, H, W] prior of anchor (detached internally)
        D_a: [1, H, W] depth of anchor
        P_b: [1, H, W] prior of neighbor (no_grad)
        D_b: [1, H, W] depth of neighbor (no_grad)
        illu_a: [3, H, W] illumination of anchor (has gradients)
        noise_a: [3, H, W] noise of anchor (has gradients)
        cam_a, cam_b: Camera objects
        alpha: weight of noise gradient in degradation gradient

    Returns:
        scalar loss
    """
    # Structure gradient of neighbor (no grad needed)
    with torch.no_grad():
        S_b = _sobel_gradient_mag(P_b)  # [1, H, W]

    # Warp S_b from view b into view a (reuses same warping as L_xv)
    S_b_warped, mask = warp_prior(D_a, S_b, D_b, cam_a, cam_b)
    S_b_warped = S_b_warped.detach()
    mask = mask.detach()

    # Minimum coverage check
    coverage = mask.sum() / mask.numel()
    if coverage < 0.05:
        return illu_a.sum() * 0.0  # zero with grad connectivity

    # Degradation gradient of anchor (has gradients through illu_a, noise_a)
    illu_mean = illu_a.mean(dim=0, keepdim=True)  # [1, H, W]
    noise_norm = noise_a.norm(dim=0, keepdim=True)  # [1, H, W]
    G_a = _sobel_gradient_mag(illu_mean) + alpha * _sobel_gradient_mag(noise_norm)  # [1, H, W]

    # Normalized inner product: <M * W(S_b), G_a> / ||M * W(S_b)||_1
    weighted_S = mask * S_b_warped
    numerator = (weighted_S * G_a).sum()
    denominator = weighted_S.abs().sum() + 1e-8
    loss = numerator / denominator

    return loss


def _ssim_map(x, y, window_size=11):
    """Compute per-pixel SSIM map. x, y: [B, C, H, W]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    # Uniform window (simple box filter)
    window = torch.ones(1, 1, window_size, window_size, device=x.device, dtype=x.dtype)
    window = window / (window_size * window_size)

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map
