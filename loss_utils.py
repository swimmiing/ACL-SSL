import torch
import torch.nn.functional as F


def infonce(pred: torch.Tensor, target: torch.Tensor, beta: float = 1/0.07, **kwargs) -> torch.Tensor:
    '''
    Compute the InfoNCE (Noise Contrastive Estimation) loss.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.
        beta (float, optional): Temperature parameter. Default is 1/0.07.

    Returns:
        torch.Tensor: InfoNCE loss.
    '''
    B = pred.shape[0]
    logits = torch.einsum('nc,mc->nm', F.normalize(pred), F.normalize(target)) * beta
    labels = torch.arange(B).long().to(pred.device)
    loss = F.cross_entropy(logits, labels)

    return loss


def area_reg(p_area: torch.Tensor, n_area: torch.Tensor, p_thr: float = 0.4, n_thr: float = 0.0,
             **kwargs) -> torch.Tensor:
    '''
    Compute the area regularization loss.

    Args:
        p_area (torch.Tensor): Positive area tensor.
        n_area (torch.Tensor): Negative area tensor.
        p_thr (float, optional): Expected positive area. Default is 0.4.
        n_thr (float, optional): Expected negative area. Default is 0.0.

    Returns:
        torch.Tensor: Area regularization loss.
    '''
    loss = torch.abs(p_area - p_thr) + torch.abs(n_area - n_thr)
    return loss


def acl_i(v_i: torch.Tensor, pred_emb: torch.Tensor, beta: float = 1 / 0.07, **kwargs) -> torch.Tensor:
    '''
    Compute the image-level audio-grounded contrastive learning (ACL_I) loss.

    Args:
        v_i (torch.Tensor): Image-level audio-grounded visual embedding tensor.
        pred_emb (torch.Tensor): Audio-driven embedding tensor.
        beta (float, optional): Temperature parameter. Default is 1/0.07.

    Returns:
        torch.Tensor: Image-level ACL loss
    '''
    loss = 0.5 * (infonce(pred_emb, v_i, beta=beta) + infonce(v_i, pred_emb, beta=beta))

    return loss


def acl_f(v_f: torch.Tensor, pred_emb: torch.Tensor, beta: float = 1 / 0.07, **kwargs) -> torch.Tensor:
    '''
    Compute the feature-level audio-grounded contrastive learning (ACL_F) loss.

    Args:
        v_f (torch.Tensor): Feature-level audio-grounded visual embedding tensor.
        pred_emb (torch.Tensor): Audio-driven embedding tensor.
        beta (float, optional): Temperature parameter. Default is 1/0.07.

    Returns:
        torch.Tensor: Feature-level ACL loss
    '''
    B, _, C = v_f.size()
    logits = torch.einsum('bnc,bc ->bn', F.normalize(v_f, dim=2), F.normalize(pred_emb))

    labels = torch.arange(B).long().to(pred_emb.device)
    loss = 0.5 * (F.cross_entropy(logits * beta, labels) + F.cross_entropy(logits.T * beta, labels))

    return loss
