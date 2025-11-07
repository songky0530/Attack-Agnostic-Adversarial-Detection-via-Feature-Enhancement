import os
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import pickle
import argparse
from tqdm import tqdm

from custom_utils import get_model, compute_gaussian_newton, matrix_modulus
from custom_models import VGG16_splitted, ResNet18_splitted  # 新增

def get_benign_images(dataset, split):
    if split == 'train':
        if dataset == 'CIFAR10':
            img_set = torchvision.datasets.CIFAR10(root='../cifar-data', train=True, download=True)
        elif dataset == 'CIFAR100':
            img_set = torchvision.datasets.CIFAR100(root='../cifar100-data', train=True, download=True)
        elif dataset == 'SVHN':
            img_set = torchvision.datasets.SVHN(root='../svhn-data', split='train', download=True)
        else:
            raise NotImplementedError(f'Unknown dataset: {dataset}')
    elif split == 'test':
        if dataset == 'CIFAR10':
            img_set = torchvision.datasets.CIFAR10(root='../cifar-data', train=False, download=True)
        elif dataset == 'CIFAR100':
            img_set = torchvision.datasets.CIFAR100(root='../cifar100-data', train=False, download=True)
        elif dataset == 'SVHN':
            img_set = torchvision.datasets.SVHN(root='../svhn-data', split='test', download=True)
        else:
            raise NotImplementedError(f'Unknown dataset: {dataset}')
    else:
        raise NotImplementedError(f'Unknown split: {split}')
    return img_set

def load_model(net, dataset, n_classes):
    folder = os.path.join('clean_train', dataset, net)
    model_path = os.path.join(folder, 'epoch_120.pth.tar')
    model = get_model(net, n_classes=n_classes)
    sd = torch.load(model_path, map_location='cpu')
    # 常见 checkpoint 结构容错
    state_dict = sd.get('state_dict', sd)
    model.load_state_dict(state_dict, strict=False)
    return model

def _to_chw_tensor(this_image_np):
    """(N,H,W,3) or (H,W,3) -> (N,3,H,W)"""
    if this_image_np.ndim == 3:
        this_image_np = this_image_np[None, ...]
    if this_image_np.shape[-1] == 3:
        this_image_np = this_image_np.transpose(0, 3, 1, 2)
    return this_image_np

def compute_gaussian_newton_features(model, images, save_path, modulus_mode='l1', device=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Resume if checkpoint exists
    if os.path.exists(save_path):
        with open(save_path, 'rb') as IN:
            res = pickle.load(IN)
        feature = res['feature']
        start_idx = res['num_processed']
        print('Resumed from idx', start_idx)
    else:
        start_idx = 0
        # Determine feature length dynamically using first image
        probe_np = _to_chw_tensor(images[0:1].astype(np.float32))
        probe_t = torch.tensor(probe_np, requires_grad=True).contiguous().to(device)
        with torch.no_grad():
            pass
        feat_vec = model.compute_layerwise_gaussian_newton(probe_t, [0], modulus_mode)
        feat_vec = np.array(feat_vec, dtype=np.float32).reshape(-1)
        # Add 1 extra dimension for the direction-sensitivity feature
        feature_len = int(feat_vec.shape[0]) + 1
        feature = np.zeros((len(images), feature_len), dtype=np.float32)

    for idx in tqdm(range(start_idx, len(images))):
        this_image_np = _to_chw_tensor(images[idx: idx+1].astype(np.float32))
        this_image_t = torch.tensor(this_image_np, requires_grad=True).contiguous().to(device)

        # Compute original Gaussian-Newton feature vector
        vec = model.compute_layerwise_gaussian_newton(this_image_t, [0], modulus_mode)
        vec = np.array(vec, dtype=np.float32).reshape(-1)

        # Compute a simple direction-sensitivity feature using Sobel gradients on the input
        # Convert image to grayscale (average over RGB channels)
        gray = torch.mean(this_image_t, dim=1, keepdim=True)  # shape (1,1,H,W)
        # Define Sobel X and Y kernels
        sobel_x = torch.tensor([[1.0, 0.0, -1.0],
                                [2.0, 0.0, -2.0],
                                [1.0, 0.0, -1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1.0,  2.0,  1.0],
                                [0.0,  0.0,  0.0],
                                [-1.0, -2.0, -1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        # Convolve with Sobel kernels
        grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        # Sum of absolute gradients in each direction
        sx = grad_x.abs().sum()
        sy = grad_y.abs().sum()
        # Direction feature: normalized difference between horizontal and vertical edge energy
        dir_feature = (sx - sy) / (sx + sy + 1e-6)
        dir_val = dir_feature.item()

        # Append direction feature to the original feature vector
        vec = np.concatenate((vec, np.array([dir_val], dtype=np.float32)), axis=0)
        feature[idx] = vec

        if (idx + 1) % 100 == 0:
            with open(save_path, 'wb') as OUT:
                pickle.dump({'feature': feature, 'num_processed': idx + 1}, OUT)

    # Save final feature array
    with open(save_path, 'wb') as OUT:
        pickle.dump({'feature': feature, 'num_processed': len(images)}, OUT)


parser = argparse.ArgumentParser()
parser.add_argument('--adv_type', type=str, default='CW')
parser.add_argument('--net', type=str, default='resnet18')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--modulus_mode', type=str, default='l1')
parser.add_argument('--out_dir', type=str, default='precomputed_GGN_modulus')
args = parser.parse_args()

# 1. arguments
adv_type = args.adv_type
net = args.net
dataset = args.dataset
split = args.split
modulus_mode = args.modulus_mode
out_dir = args.out_dir

save_path = f'{out_dir}/multi_modulus/{dataset}/{net}_{adv_type}_{split}.pkl'

if dataset == 'CIFAR100':
    n_classes = 100
else:
    n_classes = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gn_model = load_model(net, dataset, n_classes).to(device).eval()
if net.lower() == 'vgg16':
    gn_model = VGG16_splitted(gn_model).to(device).eval()
elif net.lower() == 'resnet18':
    gn_model = ResNet18_splitted(gn_model).to(device).eval()
else:
    raise NotImplementedError(f'Unknown net: {net}')

# 2. load test clean image and adversarial image
if adv_type == 'benign':
    img_set = get_benign_images(dataset, split)
    # CIFAR 与 SVHN 默认 .data 为 (N,H,W,3)
    if hasattr(img_set, 'data'):
        images = (img_set.data.astype(np.float32) / 255.0)
    elif hasattr(img_set, 'images'):  # 兼容 SVHN
        images = (np.transpose(img_set.images, (0, 2, 3, 1)).astype(np.float32) / 255.0)
    else:
        raise RuntimeError('Unknown image attribute in dataset')
else:
    # adv images
    adv_file_path = f'precomputed_adv_images_cross_model/{dataset}/{net}_{adv_type}_{split}.pkl'
    with open(adv_file_path, 'rb') as IN:
        res = pickle.load(IN)
        images = res['adv_images']
        num_valid_images = res['num_processed']
        if num_valid_images != len(images):
            # make sure all adv images are generated
            raise RuntimeError('Wait for {} to complete'.format(adv_file_path))

# 3. precompute gaussian newton features
compute_gaussian_newton_features(
    gn_model, images, save_path,
    modulus_mode=modulus_mode, device=device
)
