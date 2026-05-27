import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from math import exp

# 修改SSIM导入，兼容旧版本torchmetrics
try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
except ImportError:
    # 如果导入失败，使用自定义SSIM实现
    def ssim(pred, target, data_range=1.0):
        # 简单的SSIM实现
        mu1 = pred.mean()
        mu2 = target.mean()
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = pred.var()
        sigma2_sq = target.var()
        sigma12 = ((pred - mu1) * (target - mu2)).mean()
        
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map

def gaussian(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2))
                        for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma=1.5):
    """创建二维高斯窗口"""
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    return _2D_window.unsqueeze(0).unsqueeze(0)

def custom_ssim(pred, target, window_size=11, size_average=True, data_range=1.0):
    """自定义SSIM实现，支持多通道图像，添加数值稳定性检查"""
    # 确保输入是4D张量
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 检查输入是否包含nan或inf
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("警告: pred包含nan或inf值")
        return torch.tensor(0.0, device=pred.device)
    if torch.isnan(target).any() or torch.isinf(target).any():
        print("警告: target包含nan或inf值")
        return torch.tensor(0.0, device=target.device)
    
    # 处理多通道图像：对每个通道分别计算SSIM，然后平均
    B, C, H, W = pred.shape
    ssim_values = []
    
    for c in range(C):
        pred_c = pred[:, c:c+1]  # [B, 1, H, W]
        target_c = target[:, c:c+1]  # [B, 1, H, W]
        
        # 确保数据在合理范围内
        pred_c = torch.clamp(pred_c, 0.0, data_range)
        target_c = torch.clamp(target_c, 0.0, data_range)
        
        window = create_window(window_size).to(pred.device)
        
        mu1 = F.conv2d(pred_c, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(target_c, window, padding=window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred_c * pred_c, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(target_c * target_c, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred_c * target_c, window, padding=window_size//2, groups=1) - mu1_mu2
        
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # 添加数值稳定性检查
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        
        # 避免除零
        epsilon = 1e-6
        denominator = torch.clamp(denominator, min=epsilon)
        
        ssim_map = numerator / denominator
        
        # 添加裁剪操作，确保SSIM值不会超出理论范围[0,1]
        ssim_map = torch.clamp(ssim_map, 0.0, 1.0)
        
        # 检查计算结果
        if torch.isnan(ssim_map).any() or torch.isinf(ssim_map).any():
            print(f"警告: 通道{c}的SSIM计算出现nan/inf，使用默认值")
            ssim_map = torch.ones_like(ssim_map) * 0.5  # 使用默认值
        
        if size_average:
            ssim_values.append(ssim_map.mean())
        else:
            ssim_values.append(ssim_map.mean(1).mean(1).mean(1))
    
    # 对所有通道的SSIM值取平均
    result = torch.stack(ssim_values).mean()
    
    # 最终检查
    if torch.isnan(result) or torch.isinf(result):
        print("警告: SSIM最终结果出现nan/inf，使用默认值")
        return torch.tensor(0.5, device=pred.device)
    
    return result

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self, pred, target):
        # 使用自定义SSIM实现
        ssim_val = custom_ssim(pred, target, data_range=1.0)
        loss = 1.0 - ssim_val
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        else:
            return self.loss_weight * loss

# 移除自定义MSELoss和SSIMLoss，只保留KD等自定义损失

@LOSS_REGISTRY.register()
class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.15):
        super(KDLoss, self).__init__()
    
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach()/ self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                        S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs
                