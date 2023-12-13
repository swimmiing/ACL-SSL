import torch
import numpy as np
from sklearn import metrics
from typing import List, Optional, Tuple, Dict


class Evaluator(object):
    def __init__(self) -> None:
        """
        Initialize the Flickr Evaluator.

        Attributes:
            ciou (List[float]): Buffer of cIoU values.
            AUC (List[float]): Buffer of AUC values.
            N (int): Counter for the number of evaluations.
            metrics (List[str]): List of metric names.
        """
        super(Evaluator, self).__init__()
        self.ciou = []
        self.AUC = []
        self.N = 0
        self.metrics = ['cIoU', 'AUC']

    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor, thr: Optional[float] = None) -> None:
        """
        Evaluate a batch of predictions against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth maps.
            thr (Optional[float]): Threshold for binary classification. If None, dynamically determined.

        Returns:
            None
        """
        for j in range(pred.size(0)):
            infer = pred[j]
            gt = target[j]
            if thr is None:
                thr = np.sort(infer.detach().cpu().numpy().flatten())[int(infer.shape[1] * infer.shape[2] / 2)]
            self.cal_CIOU(infer, gt, thr)

    def cal_CIOU(self, infer: torch.Tensor, gtmap: torch.Tensor, thres: float = 0.01) -> List[float]:
        """
        Calculate cIoU (consensus Intersection over Union).

        Args:
            infer (torch.Tensor): Model prediction.
            gtmap (torch.Tensor): Ground truth map.
            thres (float): Threshold for binary classification.

        Returns:
            List[float]: List of cIoU values for each instance in the batch.
        """
        infer_map = torch.zeros_like(gtmap)
        infer_map[infer >= thres] = 1
        ciou = (infer_map * gtmap).sum(2).sum(1) / (gtmap.sum(2).sum(1) + (infer_map * (gtmap == 0)).sum(2).sum(1))

        for i in range(gtmap.size(0)):
            self.ciou.append(ciou[i].detach().cpu())
        return ciou

    def finalize_AUC(self) -> float:
        """
        Calculate the Area Under the Curve (AUC).

        Returns:
            float: AUC value.
        """
        cious = [np.sum(np.array(self.ciou) >= 0.05 * i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05 * i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self) -> float:
        """
        Calculate Average Precision (cIoU@0.5).

        Returns:
            float: cIoU@0.5 value.
        """
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self) -> float:
        """
        Calculate mean cIoU.

        Returns:
            float: Mean cIoU value.
        """
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def finalize(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Finalize and return evaluation metrics.

        Returns:
            Tuple[List[str], Dict[str, float]]: List of metric names and corresponding values.
        """
        ap50 = self.finalize_AP50() * 100
        auc = self.finalize_AUC() * 100
        return self.metrics, {self.metrics[0]: ap50, self.metrics[1]: auc}
