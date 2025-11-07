# cam_filter.py
import os
import numpy as np
import torch
import torch.nn.functional as F

# ------- Grad-CAM 核心 -------
@torch.no_grad()
def _get_pred_class(model, x):
    logits = model(x)
    return torch.argmax(logits, dim=1)

@torch.no_grad()
def get_multi_layer_gradcam(model, x, target_layers):
    """
    获取多层 Grad-CAM 并融合。返回 [N,1,H,W] 热图。
    """
    model.eval()
    feats = {}
    grads = {}

    handles = []
    for i, layer in enumerate(target_layers):
        feats[i] = None
        grads[i] = None

        def save_fwd(idx):
            return lambda m, inp, out: feats.__setitem__(idx, out)

        def save_bwd(idx):
            return lambda m, gin, gout: grads.__setitem__(idx, gout[0])

        handles.append(layer.register_forward_hook(save_fwd(i)))
        handles.append(layer.register_full_backward_hook(save_bwd(i)))

    with torch.enable_grad():
        x.requires_grad_(True)
        out = model(x)
        pred = torch.argmax(out, dim=1)
        sel = out[torch.arange(out.shape[0]), pred]
        model.zero_grad(set_to_none=True)
        sel.sum().backward()

    cams = []
    for i in feats:
        fm = feats[i]     # [N,C,h,w]
        gd = grads[i]     # [N,C,h,w]
        w = gd.mean(dim=(2,3), keepdim=True)  # [N,C,1,1]
        cam = F.relu((w * fm).sum(dim=1, keepdim=True))  # [N,1,h,w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cams.append(cam)

    # 多层热图平均融合
    heatmap = torch.stack(cams).mean(dim=0)  # [N,1,H,W]
    heatmap = heatmap - heatmap.amin(dim=(2,3), keepdim=True)
    heatmap = heatmap / (heatmap.amax(dim=(2,3), keepdim=True) + 1e-8)

    for h in handles:
        h.remove()

    return heatmap.detach()

# ------- 频域 Butterworth 高通 -------
def _butterworth_hp_mask(h, w, d0, n, device):
    # d0 以像素为单位；n 为阶数
    u = torch.arange(h, device=device) - h/2
    v = torch.arange(w, device=device) - w/2
    V, U = torch.meshgrid(v, u, indexing='ij')  # [w,h]
    D = torch.sqrt(U.T**2 + V.T**2)             # [h,w]
    H = 1.0 / (1.0 + (d0 / (D + 1e-8))**(2*n))
    return H

def apply_cam_weighted_highpass(x, heat, d0=6.0, order=2, gamma=1.0, alpha=1.0):
    """
    x:     [N,3,H,W] float tensor 0~1
    heat:  [N,1,H,W] 0~1 (Grad-CAM)
    d0:    截止频率(像素)
    order: 阶数
    gamma: CAM 权重指数（>1 会减小热图权重）
    alpha: 滤波强度增益（=1 基线；<1 更弱；>1 更强）
    return: processed [N,3,H,W]
    """
    N, C, H, W = x.shape
    device = x.device
    Hmask = _butterworth_hp_mask(H, W, d0, order, device)  # [H,W]
    # 频域滤波得到高频分量
    X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2,-1)), dim=(-2,-1))
    Xhp = X * Hmask  # 广播到 [N,3,H,W]
    xhp = torch.fft.ifft2(torch.fft.ifftshift(Xhp, dim=(-2,-1)), dim=(-2,-1)).real

    # 权重映射 + 强度
    w = heat.clip(0,1)**gamma              # [N,1,H,W]
    x_out = x + alpha * xhp * w            # 叠加高频到关注区域（含强度 alpha）
    return x_out.clamp(0.0, 1.0)

# ------- 复杂度指标：频域熵（Fourier entropy） -------
def _fourier_entropy_np(img, eps=1e-12):
    """
    img: (H,W,3) in [0,1] numpy
    return: Shannon entropy of magnitude spectrum (越大=越复杂)
    """
    gray = np.mean(img, axis=2)
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)                         # 幅度
    p = mag.ravel().astype(np.float64)
    p = p / (p.sum() + eps)                 # 概率分布
    return float(-(p * np.log(p + eps)).sum())

# ------- 一站式接口（供外部调用） -------
def cam_highpass_process(
    images_np,
    model,
    target_layer,
    d0=6.0,
    order=2,
    gamma=1.0,              # 静态路径/基线的 CAM gamma
    batch_size=128,
    adaptive_gamma=False,   # 开关：是否启用基于熵的自适应
    # 自适应分档(频域熵)的固定阈值与档位参数（可按需修改）
    entropy_thr=5.5632,     # 固定阈值（来自你统计）5.5632,
    alpha_weak=1.,         # 熵 < thr  -> 滤波弱
    alpha_strong=1.,         # 熵 ≥ thr  -> 滤波强
    gamma_low=1.,          # 熵 ≥ thr  -> CAM gamma 小（权重大）
    gamma_high=1.          # 熵 < thr  -> CAM gamma 大（权重小）
):
    """
    images_np: (N,H,W,3) numpy, [0,1]
    adaptive_gamma:
        - False: 使用静态 gamma 和 alpha=1（不改强度）
        - True : 同时按频域熵固定阈值分档【alpha 滤波强度】与【gamma CAM 权重指数】
                 熵 < thr  -> alpha_weak,  gamma_high
                 熵 >= thr -> alpha_strong,gamma_low
    返回同尺寸 numpy
    """
    assert images_np.ndim == 4 and images_np.shape[-1] == 3
    device = next(model.parameters()).device
    N, H, W, _ = images_np.shape
    outs = []

    # 提前构建一次频域掩码（用于自适应分支的频域高通）
    if adaptive_gamma:
        Hmask = _butterworth_hp_mask(H, W, d0, order, device)  # [H,W]

    with torch.no_grad():
        for i in range(0, N, batch_size):
            chunk = images_np[i:i+batch_size]                                  # (B,H,W,3) numpy
            t = torch.from_numpy(chunk).permute(0,3,1,2).to(device).float()     # [B,3,H,W]

            # Grad-CAM（需开启梯度）
            torch.set_grad_enabled(True)
            heat = get_multi_layer_gradcam(model, t, [model.layer2[-1], model.layer3[-1], model.layer4[-1]])
            torch.set_grad_enabled(False)

            if not adaptive_gamma:
                # 静态路径：alpha=1，仅使用传入的 gamma
                proc = apply_cam_weighted_highpass(t, heat, d0=d0, order=order, gamma=gamma, alpha=1.0)
            else:
                # ========= 频域熵 → (alpha, gamma)（固定单阈值，两档）=========
                entropies = [_fourier_entropy_np(img) for img in chunk]  # list[float], len=B

                # 逐图分档
                alpha_vals = []
                gamma_vals = []
                for s in entropies:
                    if s < entropy_thr:
                        # 简单图：滤波弱，权重更小
                        alpha_vals.append(alpha_weak)
                        gamma_vals.append(gamma_high)
                    else:
                        # 复杂图：滤波强，权重更大
                        alpha_vals.append(alpha_strong)
                        gamma_vals.append(gamma_low)

                alpha_tensor = torch.tensor(alpha_vals, device=device, dtype=torch.float32).view(-1,1,1,1)
                gamma_tensor = torch.tensor(gamma_vals, device=device, dtype=torch.float32).view(-1,1,1,1)

                # 频域 Butterworth 高通
                X = torch.fft.fftshift(torch.fft.fft2(t, dim=(-2,-1)), dim=(-2,-1))
                Xhp = X * Hmask  # [B,3,H,W]
                xhp = torch.fft.ifft2(torch.fft.ifftshift(Xhp, dim=(-2,-1)), dim=(-2,-1)).real

                # 权重映射与叠加：alpha 控滤波强度；gamma 控 CAM 权重指数
                w = heat.clip(0,1) ** gamma_tensor   # [B,1,H,W]
                x_out = t + alpha_tensor * xhp * w
                proc = x_out.clamp(0.0, 1.0)

            outs.append(proc.permute(0,2,3,1).cpu().numpy())

    return np.concatenate(outs, axis=0)
