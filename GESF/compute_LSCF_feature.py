# compute_LSCF_feature.py
import os
import numpy as np
from sklearn.decomposition import PCA
import torchvision
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--adv_type', type=str, default='benign')
parser.add_argument('--net', type=str, default='resnet18')
parser.add_argument('--feat_dim', type=int, default=32)
# ===== 新增：Grad-CAM + 加权高通滤波 开关与参数 =====
parser.add_argument('--use_cam', type=int, default=1, help='1=启用 Grad-CAM 引导高通后再提 LSCF')
parser.add_argument('--cam_d0', type=float, default=12.0, help='Butterworth 截止频率(像素)')
parser.add_argument('--cam_order', type=int, default=2, help='Butterworth 滤波器阶数')
parser.add_argument('--cam_gamma', type=float, default=2.0, help='权重指数映射（>1 更强调高权重区域）')
parser.add_argument('--adaptive_gamma', type=int, default=1, help='0=关闭动态 gamma 调整, 1=启用动态 gamma 调整')
args = parser.parse_args()

split = args.split
dataset = args.dataset
adv_type = args.adv_type
net = args.net
feat_dim = args.feat_dim

# 原有 PCA 模型
model_path = f'precomputed_LSCF_model/{dataset}_train.pkl'

# 保存路径：默认原目录；若 use_cam=1，写到 _cam 目录避免覆盖
if args.use_cam == 1:
    save_path = f'precomputed_LSCF_feature_cam/{dataset}/{net}_{adv_type}_{split}.pkl'
else:
    save_path = f'precomputed_LSCF_feature/{dataset}/{net}_{adv_type}_{split}.pkl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ===== 读取图像（保持你原有逻辑） =====
if adv_type == 'benign':
    # 加载官方数据集
    if split == 'train':
        if dataset == 'CIFAR10':
            data = torchvision.datasets.CIFAR10(root='./cifar-data', train=True, download=True)
        elif dataset == 'CIFAR100':
            data = torchvision.datasets.CIFAR100(root='./cifar100-data', train=True, download=True)
        elif dataset == 'SVHN':
            data = torchvision.datasets.SVHN(root='./svhn-data', split='train', download=True)
    if split == 'test':
        if dataset == 'CIFAR10':
            data = torchvision.datasets.CIFAR10(root='./cifar-data', train=False, download=True)
        elif dataset == 'CIFAR100':
            data = torchvision.datasets.CIFAR100(root='./cifar100-data', train=False, download=True)
        elif dataset == 'SVHN':
            data = torchvision.datasets.SVHN(root='./svhn-data', split='test', download=True)

    images = data.data                        # (N,H,W,3)
    if images.shape[-1] == 3:
        images = images.transpose(0, 3, 1, 2) # -> (N,3,H,W)
    images = images / 255.0                   # 保持与 PCA 训练时一致
else:
    adv_file_path = f'precomputed_adv_images_cross_model/{dataset}/{net}_{adv_type}_{split}.pkl'
    with open(adv_file_path, 'rb') as IN:
        res = pickle.load(IN)
    images = res['adv_images']                # (N,H,W,3) 或 [0,1]/[0,255]
    images = images.transpose(0, 3, 1, 2)     # -> (N,3,H,W)
    # 保证范围为 [0,1]
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0

# ====== 在 reshape 之前：可选 CAM+高通滤波 ======
if args.use_cam == 1:
    import torch
    from cam_filter import cam_highpass_process
    try:
        # 这里假定你在 custom_utils.py 里有 get_model(net, n_classes)
        from custom_utils import get_model
    except ImportError:
        raise ImportError("需要 custom_utils.get_model 来加载分类模型。")

    n_classes = 100 if dataset == 'CIFAR100' else 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(net, n_classes=n_classes).to(device).eval()

    # 尝试从常见路径加载权重（可按你项目实际调整；未找到也能跑 Grad-CAM 但效果会受影响）
    ckpt_candidates = [
        os.path.join('clean_train', dataset, net, 'best.pth.tar'),
        os.path.join('clean_train', dataset, net, 'epoch_120.pth.tar'),
        os.path.join('checkpoints', dataset, f'{net}.pth'),
    ]
    for ck in ckpt_candidates:
        if os.path.exists(ck):
            sd = torch.load(ck, map_location='cpu')
            state_dict = sd.get('state_dict', sd)
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
            break

    # 统一到 (N,H,W,3) float32 [0,1]
    images_np = images.transpose(0, 2, 3, 1).astype(np.float32)

    # 选择 ResNet18 的最后卷积块作为 target_layer（VGG 请自行调整）
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    else:
        raise RuntimeError('未找到 model.layer4；请根据你的网络结构修改 target_layer')

    # 运行 CAM+高通
    if args.adaptive_gamma == 1:
        images_np = cam_highpass_process(
            images_np, model, target_layer,
            d0=args.cam_d0, order=args.cam_order, gamma=args.cam_gamma, batch_size=256,
            adaptive_gamma=True
        )
    else:
        images_np = cam_highpass_process(
            images_np, model, target_layer,
            d0=args.cam_d0, order=args.cam_order, gamma=args.cam_gamma, batch_size=256
        )
    # 回到 (N,3,H,W)
    images = images_np.transpose(0, 3, 1, 2)

num_samples = images.shape[0]
images = images.reshape(num_samples, -1)

with open(model_path, 'rb') as IN:
    pca_model = pickle.load(IN)

images_feat = pca_model.transform(images)

with open(save_path, 'wb') as OUT:
    pickle.dump({
        'major': images_feat[:, :feat_dim],
        'minor': images_feat[:, -feat_dim:]
    }, OUT)

print('[OK] Saved:', os.path.abspath(save_path))
