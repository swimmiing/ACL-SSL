import torch
import numpy as np
from typing import List, Tuple, Dict


class Evaluator(object):
    def __init__(self) -> None:
        """
        Initialize the AVSBench evaluator.

        Attributes:
            miou (List[float]): Buffer of mIoU values.
            F (List[float]): Buffer of F-measure values.
            N (int): Counter for the number of evaluations.
            metrics (List[str]): List of metric names.
        """
        super(Evaluator, self).__init__()
        self.miou = []
        self.F = []
        self.N = 0
        self.metrics = ['mIoU', 'Fmeasure']

    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor, thr: List[float] = None) -> None:
        """
        Evaluate a batch of predictions against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth.
            thr (List[float], optional): List of thresholds. If None, calculate threshold as median. Default is None.

        Notes:
            Updates metric buffers (self.mask_iou, self.Eval_Fmeasusre)
        """
        thrs = []

        for j in range(pred.size(0)):
            infer = pred[j]
            if thr is None:
                thrs.append(np.sort(infer.detach().cpu().numpy().flatten())[int(infer.shape[1] * infer.shape[2] / 2)])
            else:
                thrs.append(thr)

        infers, gts = pred.squeeze(1), target.squeeze(1)
        self.mask_iou(infers, gts, thrs)
        self.Eval_Fmeasure(infers, gts)

    def mask_iou(self, preds: torch.Tensor, targets: torch.Tensor, thrs: List[float], eps: float = 1e-7) -> float:
        """
        Calculate mask IoU.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth.
            thrs (List[float]): List of thresholds.
            eps (float, optional): Small epsilon to avoid division by zero. Default is 1e-7.

        Returns:
            float: mIoU value.
        """
        assert len(preds.shape) == 3 and preds.shape == targets.shape
        self.N += 1

        N = preds.size(0)
        miou = 0.0
        for i in range(N):
            pred = preds[i].unsqueeze(0)
            target = targets[i].unsqueeze(0)

            num_pixels = pred.size(-1) * pred.size(-2)
            no_obj_flag = (target.sum(2).sum(1) == 0)

            pred = (pred > thrs[i]).int()
            inter = (pred * target).sum(2).sum(1)
            union = torch.max(pred, target).sum(2).sum(1)

            inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
            inter[no_obj_flag] = inter_no_obj[no_obj_flag]
            union[no_obj_flag] = num_pixels
            miou += (torch.sum(inter / (union + eps))).squeeze()
        miou = miou / N
        self.miou.append(miou.detach().cpu())

        return miou

    @staticmethod
    def _eval_pr(y_pred: torch.Tensor, y: torch.Tensor, num: int, cuda_flag: bool = True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate precision and recall.

        Args:
            y_pred (torch.Tensor): Model predictions.
            y (torch.Tensor): Ground truth.
            num (int): Number of threshold values.
            cuda_flag (bool, optional): Whether to use CUDA. Default is True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Precision and recall values.
        """
        if cuda_flag:
            prec, recall = torch.zeros(num).to(y_pred.device), torch.zeros(num).to(y_pred.device)
            thlist = torch.linspace(0, 1 - 1e-10, num).to(y_pred.device)
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

        return prec, recall

    def Eval_Fmeasure(self, pred: torch.Tensor, gt: torch.Tensor, pr_num: int = 255) -> float:
        """
        Evaluate F-measure.

        Args:
            pred (torch.Tensor): Model predictions.
            gt (torch.Tensor): Ground truth.
            pr_num (int, optional): Number of precision-recall values. Default is 255.

        Returns:
            float: F-measure value.

        Notes:
            Fix bug in official test code (Issue: Results vary depending on the batch number)
            The official code had an issue because it optimized precision-recall thresholds for each mini-batch.
        """
        N = pred.size(0)
        beta2 = 0.3
        avg_f, img_num = 0.0, 0
        score = torch.zeros(pr_num).to(pred.device)

        for img_id in range(N):
            # examples with totally black GTs are out of consideration
            if torch.sum(gt[img_id]) == 0.0:
                continue
            prec, recall = self._eval_pr(pred[img_id], gt[img_id], pr_num)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for Nan
            avg_f += f_score
            img_num += 1
            score = avg_f / img_num
            self.F.append(f_score.detach().cpu().numpy())
            # print('score: ', score)

        return score.max().item()

    def finalize_mIoU(self) -> float:
        """
        Calculate the final mIoU value.

        Returns:
            float: Final mIoU value.
        """
        miou = np.sum(np.array(self.miou)) / self.N
        return miou

    def finalize_Fmeasure(self) -> float:
        """
        Calculate the final F-measure value.

        Returns:
            float: Final F-measure value.

        Notes:
            Fix bug in official test code (Issue: Results vary depending on the batch number)
            The official code had an issue because it optimized precision-recall thresholds for each mini-batch
        """
        # F = np.sum(np.array(self.F)) / self.N
        F = np.max(np.mean(self.F, axis=0))

        return F

    def finalize(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Finalize evaluation and return the results.

        Returns:
            Tuple[List[str], Dict[str, float]]: Tuple containing metric names and corresponding values.
        """
        mIoU = self.finalize_mIoU() * 100
        F = self.finalize_Fmeasure() * 100
        return self.metrics, {self.metrics[0]: mIoU, self.metrics[1]: F}
