from copy import deepcopy

import random
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import numpy as np

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class Vcotta(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, class_num=10, batch_size=200):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.class_num = class_num
        self.batch_size = batch_size

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
    
        mix_factor =  torch.rand(1).cuda()

        idx = torch.randperm(x.size(0))
        shuffled_x = x[idx] # shuffle data
        mix_x = mix_factor * x + (1 - mix_factor) * shuffled_x # input mix
        mix_student_outputs = self.model(mix_x, sample=True) 

        outputs_anchor, outputs_anchor_entropy, outputs_ema, outputs_ema_entropy  = self.evaluate_output_uncertainty(self.model_anchor,self.model_ema, x, 32)

        T = 8e-4

        (teacher_factor, source_factor) = F.softmax(torch.tensor([outputs_ema_entropy, outputs_anchor_entropy])/T, dim=0)

        shuffled_outputs_ema = outputs_ema[idx]
        mix_teacher_outputs = mix_factor * outputs_ema + (1 - mix_factor) * shuffled_outputs_ema # output mix

        shuffled_outputs_anchor = outputs_anchor[idx]
        mix_source_outputs = mix_factor * outputs_anchor + (1 - mix_factor) * shuffled_outputs_anchor # output mix

        outputs_ref = teacher_factor * mix_teacher_outputs + source_factor * mix_source_outputs

        entropy_loss = symmetric_cross_entropy(mix_student_outputs, outputs_ref).mean(0) 

        kl_loss = teacher_factor * self.model.compute_kl_with_prior(self.model_ema)
        kl_loss += source_factor * self.model.compute_kl_with_prior(self.model_anchor)
        
        loss = entropy_loss  + kl_loss / 500000.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Teacher update, moving average
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        outputs = teacher_factor * outputs_ema + source_factor * outputs_anchor

        # return outputs, loss
        return outputs

    def add_random_perturbation(self, x, stddev):
        perturbation = torch.randn_like(x) * stddev
        perturbed_x = x + perturbation
        return perturbed_x

    def evaluate_output_uncertainty(self, model_anchor, model_ema, x, aug_num):
        outputs_anchor, outputs_ema = [], []
        output_anchor, output_ema = model_anchor(x, sample=False).detach(), model_ema(x, sample=False).detach()
        outputs_anchor.append(output_anchor)
        outputs_ema.append(output_ema)

        for _ in range(aug_num):
            x_aug = self.transform(x)
            outputs_anchor.append(model_anchor(x_aug, sample=False).detach())
            outputs_ema.append(model_ema(x_aug, sample=False).detach())

        outputs_anchor, outputs_ema = torch.stack(outputs_anchor), torch.stack(outputs_ema)
        outputs_anchor_prob, outputs_ema_prob = F.softmax(outputs_anchor, dim=2), F.softmax(outputs_ema, dim=2)

        margin = 0.3
        output_anchor_mean, outputs_entropy_entropy = self.select_outputs(outputs_anchor, outputs_anchor_prob, margin, aug_num)
        output_ema_mean, outputs_ema_entropy = self.select_outputs(outputs_ema, outputs_ema_prob, margin, aug_num)
        

        return output_anchor_mean, outputs_entropy_entropy, output_ema_mean, outputs_ema_entropy

    def select_outputs(self, outputs, outputs_prob, margin, aug_num):
        output_max = outputs_prob.max(-1)[0].detach()
        outputs_max_ind = outputs_prob.max(-1)[0] >= ((output_max + margin).unsqueeze(0).expand(aug_num+1, self.batch_size)) 
        outputs_max_ind[0, :] = True
        output_size=outputs.size()

        outputs_max_num = torch.sum(outputs_max_ind, dim=0).unsqueeze(0).unsqueeze(-1).expand(aug_num+1,self.batch_size,self.class_num)
        outputs_max_mask = outputs_max_ind.unsqueeze(2).expand(aug_num+1, self.batch_size, self.class_num)
        selected_outputs = outputs * outputs_max_mask / outputs_max_num
        outputs_prob = F.softmax(outputs, dim=2)
        outputs_entropy = torch.mean(torch.exp(outputs_prob) * outputs_prob)
        output_mean = selected_outputs.sum(0)

        return output_mean, outputs_entropy



@torch.jit.script
def softmax_shannon_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1) # Shannon entrophy

@torch.jit.script
def renyi_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x*x).sum(1).log_softmax(0) # Renyi entrophy


@torch.jit.script
def softmax_cross_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

# @torch.jit.script
def softmax_collision_cross_entropy(x, x_ema):
    """
    Collosion Cross Entropy of softmax distribution from logits.
    Zhongwen (Rex) Zhang et al., Collision Cross-entropy for Soft Class Labels and Deep Clustering, Arxiv'23
    """
    return -torch.log((x_ema.softmax(1) * x.softmax(1)).sum(1))



@torch.jit.script
def symmetric_cross_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Mario Dobler et al. RMT CVPR-23"""
    alpha = 0.5
    loss = - (1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) 
    loss += - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
    return loss



@torch.jit.script
def interpolation_consistency(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return ((x-y)**2).sum(1) # Shannon entrophy

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names

def collect_bayes_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias', 'weight_mu', 'bias_mu', 'weight_rho', 'bias_rho'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
