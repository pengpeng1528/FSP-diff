import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from FSP.models import lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from basicsr.losses import build_loss

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

@MODEL_REGISTRY.register()
class FSPS1Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(FSPS1Model, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.scaler = GradScaler()
        # 新增：初始化mse、ssim、lpips损失
        if self.is_train:
            mse_opt = self.opt['train'].get('mse_opt', None)
            ssim_opt = self.opt['train'].get('ssim_opt', None)
            # 由basicsr的build_loss自动构建
            self.cri_mse = build_loss(mse_opt).to(self.device) if mse_opt else None
            self.cri_ssim = build_loss(ssim_opt).to(self.device) if ssim_opt else None
    
    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(FSPS1Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.gt)
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # 前向：使用 AMP
        with autocast():
            self.output, _ = self.net_g(self.lq, self.gt)
        
        # 输出值域裁剪，适应目标图像可能比输入图像大的情况
        self.output = torch.clamp(self.output, -0.02, 1.05)
        
        loss_dict = OrderedDict()
        # 损失在 fp32 计算
        with autocast(enabled=False):
            l_total = torch.zeros((), device=self.output.device, dtype=torch.float32)
            pred32 = self.output.float()
            gt32 = self.gt.float()
            
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(pred32, gt32)
                l_total = l_total + l_pix
                loss_dict['l_pix'] = l_pix
                if l_pix.item() > 1000:
                    print(f"[警告] 损失值异常: {l_pix.item()}, 当前迭代: {current_iter}")
            
            # mse loss
            if self.cri_mse:
                l_mse = self.cri_mse(pred32, gt32)
                l_total = l_total + l_mse
                loss_dict['l_mse'] = l_mse
            
            # ssim loss
            if self.cri_ssim:
                l_ssim = self.cri_ssim(pred32, gt32)
                l_total = l_total + l_ssim
                loss_dict['l_ssim'] = l_ssim
            
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(pred32, gt32)
                if l_percep is not None:
                    l_total = l_total + l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total = l_total + l_style
                    loss_dict['l_style'] = l_style
        
        # 非有限保护
        if not torch.isfinite(l_total):
            print(f"[警告] 第{current_iter}步总损失出现非有限值，跳过该步更新。")
            return
        
        # AMP 正确顺序：scale->backward->unscale_->clip->step->update
        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
