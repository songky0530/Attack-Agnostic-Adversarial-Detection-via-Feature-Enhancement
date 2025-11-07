import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import hessian, jacobian

class ResNet18_splitted(nn.Module):
    def __init__(self, model, num_classes=10):
        super(ResNet18_splitted, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.forward_before_linear(x)
        out = self.model.linear(out)
        return out

    def forward_before_linear(self, x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        # 兼容不同实现中的自适应池化/固定池化
        if hasattr(F, 'adaptive_avg_pool2d'):
            out = F.adaptive_avg_pool2d(out, (1, 1))
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    # 修复：补上 self；补齐 forward_func；约束 batch=1（与上游一致）
    def compute_input_gaussian_newton(self, image, target, modulus_mode):
        x = image.clone().detach().requires_grad_(True)
        self.zero_grad()

        def forward_func(z):
            with torch.enable_grad():
                return self.forward(z)

        def loss_func(logits):
            with torch.enable_grad():
                return nn.CrossEntropyLoss(reduction="mean")(logits, target)

        logits = forward_func(x)

        # 形状： (1,num_logits,1,C,H,W)
        jacobian_logits_input = jacobian(forward_func, x)
        # 形状： (1,num_logits,1,C,H,W) -> (num_logits, C*H*W)
        bs, num_logits, _, c, h, w = jacobian_logits_input.shape
        assert bs == 1, 'please ensure batch size is 1'
        jacobian_logits_input = jacobian_logits_input.view(num_logits, c * h * w)

        # Hessian: (1,num_logits,1,num_logits) -> (num_logits, num_logits)
        logits_hessian = hessian(loss_func, logits)
        logits_hessian = logits_hessian.view(num_logits, num_logits)

        m1 = jacobian_logits_input.t() @ logits_hessian  # (C*H*W, num_logits)
        m2 = jacobian_logits_input                         # (num_logits, C*H*W)

        res = 0.0
        # 等价于 vec = m1 @ m2 的每列按模聚合（保留原有“逐列累加”的写法）
        for i in range(m2.shape[1]):
            this_m = m1 @ m2[:, i:i+1]  # (C*H*W, 1)
            if modulus_mode == 'l1':
                res = res + torch.sum(torch.abs(this_m))
            elif modulus_mode == 'l2':
                res = res + torch.sum(this_m ** 2)
            else:
                raise NotImplementedError(f'Unknown modulus_mode: {modulus_mode}')
        if modulus_mode == 'l2':
            res = torch.sqrt(res)
        return res.detach().cpu().float()

    # 修复：补上 self；函数在 logits 线性层之前/之后的两种 GGN 计算（沿用原意）
    def compute_logit_gaussian_newton(self, image, target, modulus_mode):
        x = image.clone().detach().requires_grad_(True)
        self.zero_grad()

        x_feat = self.forward_before_linear(x)

        def forward_func(feat):
            with torch.enable_grad():
                return self.model.linear(feat)

        def loss_func(logits):
            with torch.enable_grad():
                return nn.CrossEntropyLoss(reduction="mean")(logits, target)

        logits = forward_func(x_feat)

        # 形状：(1,num_logits,1,feat_dim) -> (num_logits, feat_dim)
        jacobian_logits_feat = jacobian(forward_func, x_feat)
        bs, num_logits, _, feat_dim = jacobian_logits_feat.shape
        assert bs == 1, 'please ensure batch size is 1'
        jacobian_logits_feat = jacobian_logits_feat.view(num_logits, feat_dim)

        # Hessian: (1,num_logits,1,num_logits) -> (num_logits, num_logits)
        logits_hessian = hessian(loss_func, logits)
        logits_hessian = logits_hessian.view(num_logits, num_logits)

        m1 = jacobian_logits_feat.t() @ logits_hessian  # (feat_dim, num_logits)
        m2 = jacobian_logits_feat                        # (num_logits, feat_dim)

        res = 0.0
        for i in range(m2.shape[1]):
            this_m = m1 @ m2[:, i:i+1]  # (feat_dim, 1)
            if modulus_mode == 'l1':
                res = res + torch.sum(torch.abs(this_m))
            elif modulus_mode == 'l2':
                res = res + torch.sum(this_m ** 2)
            else:
                raise NotImplementedError(f'Unknown modulus_mode: {modulus_mode}')
        if modulus_mode == 'l2':
            res = torch.sqrt(res)
        return res.detach().cpu().float()

    def compute_layerwise_gaussian_newton(self, image, target, modulus_mode='l1'):
        self.eval()
        # 兼容 list/int；确保 dtype/设备一致
        if isinstance(target, list):
            target = torch.tensor(target, dtype=torch.long, device=image.device)
        elif isinstance(target, int):
            target = torch.tensor([target], dtype=torch.long, device=image.device)
        elif torch.is_tensor(target):
            target = target.to(image.device).long()
        else:
            raise ValueError('Unsupported target type')

        gaussian_newton = []
        gaussian_newton.append(self.compute_input_gaussian_newton(image, target, modulus_mode).item())
        gaussian_newton.append(self.compute_logit_gaussian_newton(image, target, modulus_mode).item())
        return gaussian_newton

# 注：该文件仅提供 ResNet18_splitted 的“包裹”类，工厂函数按需自定义。
