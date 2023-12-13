import torch
import torch.nn as nn


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False):
    """Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    References:
        - https://github.com/yandexdataschool/gumbel_dpg/blob/master/gumbel.py
        - https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    Note:
        X - Y ~ Logistic(0,1) s.t. X, Y ~ Gumbel(0, 1).
        That is, we can implement gumbel_sigmoid using Logistic distribution.
    """
    logistic = torch.rand_like(logits)
    logistic = logistic.div_(1. - logistic).log_()  # ~Logistic(0,1)

    gumbels = (logits + logistic) / tau  # ~Logistic(logits, tau)
    y_soft = gumbels.sigmoid_()

    if hard:
        # Straight through.
        y_hard = y_soft.gt(0.5).type(y_soft.dtype)
        # gt_ break gradient flow
        #  y_hard = y_soft.gt_(0.5)  # gt_() maintain dtype, different to gt()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


class Sim2Mask(nn.Module):
    def __init__(self, init_w: float = 1.0, init_b: float = 0.0, gumbel_tau: float = 1.0, learnable: bool = True):
        """
        Sim2Mask module for generating binary masks.

        Args:
            init_w (float): Initial value for weight.
            init_b (float): Initial value for bias.
            gumbel_tau (float): Gumbel-Softmax temperature.
            learnable (bool): If True, weight and bias are learnable parameters.

        Reference:
            "Learning to Generate Text-grounded Mask for Open-world Semantic Segmentation from Only Image-Text Pairs" CVPR 2023
            - https://github.com/kakaobrain/tcl
            - https://arxiv.org/abs/2212.00785
        """
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.gumbel_tau = gumbel_tau
        self.learnable = learnable

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))
            self.b = nn.Parameter(torch.full([], float(init_b)))
        else:
            self.w = init_w
            self.b = init_b

    def forward(self, x, deterministic=False):
        logits = x * self.w + self.b

        soft_mask = torch.sigmoid(logits)
        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(logits.dtype)
        else:
            hard_mask = gumbel_sigmoid(logits, hard=True, tau=self.gumbel_tau)

        return hard_mask, soft_mask

    def extra_repr(self):
        return f'init_w={self.init_w}, init_b={self.init_b}, learnable={self.learnable}, gumbel_tau={self.gumbel_tau}'


def norm_img_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor to the range [0, 1].

    Args:
        tensor (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    vmin = tensor.amin((2, 3), keepdims=True) - 1e-7
    vmax = tensor.amax((2, 3), keepdims=True) + 1e-7
    tensor = (tensor - vmin) / (vmax - vmin)
    return tensor


class ImageMasker(Sim2Mask):
    def forward(self, x: torch.Tensor, infer: bool = False) -> torch.Tensor:
        """
        Forward pass for generating image-level binary masks.

        Args:
            x (torch.Tensor): Input tensor.
            infer (bool): True for only inference stage.

        Returns:
            torch.Tensor: Binary mask.

        Reference:
            "Can CLIP Help Sound Source Localization?" WACV 2024
            - https://arxiv.org/abs/2311.04066
        """
        if self.training or not infer:
            output = super().forward(x, False)[0]
        else:
            output = torch.sigmoid(x + self.b / self.w)
        return output


class FeatureMasker(nn.Module):
    def __init__(self, thr: float = 0.5, tau: float = 0.07):
        """
        Masker module for generating feature-level masks.

        Args:
            thr (float): Threshold for generating the mask.
            tau (float): Temperature for the sigmoid function.

        Reference:
            "Can CLIP Help Sound Source Localization?" WACV 2024
            - https://arxiv.org/abs/2311.04066
        """
        super().__init__()
        self.thr = thr
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generating feature-level masks

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Generated mask.
        """
        return torch.sigmoid((norm_img_tensor(x) - self.thr) / self.tau)
