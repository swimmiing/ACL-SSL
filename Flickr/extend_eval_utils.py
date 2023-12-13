import numpy as np
import torch
from sklearn import metrics


class Evaluator(object):
    def __init__(self, iou_thrs=(0.5, ), default_conf_thr=0.5, pred_size=0.5, pred_thr=0.5,
                 results_dir='./results'):
        """
        Initialize the Extended Flickr evaluator.

        Notes:
            Taking computation speed into consideration, it is set to output only the 'all' subset. (AP, Max-F1)
        """
        super(Evaluator, self).__init__()
        self.iou_thrs = iou_thrs
        self.default_conf_thr = default_conf_thr
        self.min_sizes = {'small': 0, 'medium': 32 ** 2, 'large': 96 ** 2, 'huge': 144 ** 2}
        self.max_sizes = {'small': 32 ** 2, 'medium': 96 ** 2, 'large': 144 ** 2, 'huge': 10000 ** 2}

        self.ciou_list = []
        self.area_list = []
        self.confidence_list = []
        self.name_list = []
        self.bb_list = []
        # self.metrics = ['AP', 'Max-F1', 'LocAcc']
        self.metrics = ['AP', 'Max-F1']

        self.results_dir = results_dir
        self.viz_save_dir = f"{results_dir}/viz_conf" + str(default_conf_thr) + "_predsize" + str(
            pred_size) + "_predthr" + str(pred_thr)
        self.results_save_dir = f"{results_dir}/results_conf" + str(default_conf_thr) + "_predsize" + str(
            pred_size) + "_predthr" + str(pred_thr)

    @staticmethod
    def calc_precision_recall(bb_list, ciou_list, confidence_list, confidence_thr, ciou_thr=0.5):
        assert len(bb_list) == len(ciou_list) == len(confidence_list)
        true_pos, false_pos, false_neg = 0, 0, 0
        for bb, ciou, confidence in zip(bb_list, ciou_list, confidence_list):
            if bb == 0:
                # no sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected
                    false_pos += 1
            else:
                # sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected...
                    if ciou >= ciou_thr:  # ...in correct place
                        true_pos += 1
                    else:  # ...in wrong place
                        false_pos += 1
                else:
                    # no sounding objects detected
                    false_neg += 1

        precision = 1. if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
        recall = 1. if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)

        return precision, recall

    def calc_ap(self, bb_list_full, ciou_list_full, confidence_list_full, iou_thr=0.5):

        assert len(bb_list_full) == len(ciou_list_full) == len(confidence_list_full)

        # for visible objects
        # ss = [i for i, bb in enumerate(bb_list_full) if bb > 0]
        # bb_list = [bb_list_full[i] for i in ss]
        # ciou_list = [ciou_list_full[i] for i in ss]
        # confidence_list = [confidence_list_full[i] for i in ss]

        precision, recall, skip_thr = [], [], max(1, len(ciou_list_full) // 200)
        for thr in np.sort(np.array(confidence_list_full))[:-1][::-skip_thr]:
            p, r = self.calc_precision_recall(bb_list_full, ciou_list_full, confidence_list_full, thr, iou_thr)
            precision.append(p)
            recall.append(r)
        precision_max = [np.max(precision[i:]) for i in range(len(precision))]
        ap = sum([precision_max[i] * (recall[i + 1] - recall[i])
                  for i in range(len(precision_max) - 1)])
        return ap

    def cal_auc(self, bb_list, ciou_list):
        ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        ciou = [ciou_list[i] for i in ss]
        cious = [np.sum(np.array(ciou) >= 0.05 * i) / len(ciou)
                 for i in range(21)]
        thr = [0.05 * i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def filter_subset(self, subset, name_list, area_list, bb_list, ciou_list, conf_list):
        if subset == 'visible':
            ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        elif subset == 'non-visible/non-audible':
            ss = [i for i, bb in enumerate(bb_list) if bb == 0]
        elif subset == 'all':
            ss = [i for i, bb in enumerate(bb_list) if bb >= 0]
        else:
            ss = [i for i, sz in enumerate(area_list)
                  if self.min_sizes[subset] <= sz < self.max_sizes[subset] and bb_list[i] > 0]

        if len(ss) == 0:
            return [], [], [], [], []

        name = [name_list[i] for i in ss]
        area = [area_list[i] for i in ss]
        bbox = [bb_list[i] for i in ss]
        ciou = [ciou_list[i] for i in ss]
        conf = [conf_list[i] for i in ss]

        return name, area, bbox, ciou, conf

    def finalize_stats(self):
        name_full_list, area_full_list, bb_full_list, ciou_full_list, confidence_full_list = self.gather_results()

        metrics = {}
        for iou_thr in self.iou_thrs:
            # for subset in ['all', 'visible']:
            for subset in ['all']:
                _, _, bb_list, ciou_list, conf_list = self.filter_subset(subset, name_full_list, area_full_list,
                                                                         bb_full_list, ciou_full_list,
                                                                         confidence_full_list)
                subset_name = f'{subset}@{int(iou_thr * 100)}' if subset is not None else f'@{int(iou_thr * 100)}'
                if len(ciou_list) == 0:
                    p, r, ap, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    p, r = self.calc_precision_recall(bb_list, ciou_list, conf_list, -1000, iou_thr)
                    ap = self.calc_ap(bb_list, ciou_list, conf_list, iou_thr)
                    auc = self.cal_auc(bb_list, ciou_list)

                    conf_thr = list(sorted(conf_list))[::max(1, len(conf_list) // 10)]
                    pr = [self.calc_precision_recall(bb_list, ciou_list, conf_list, thr, iou_thr) for thr in conf_thr]
                    f1 = [2 * r * p / (r + p) if r + p > 0 else 0. for p, r in pr]
                    if subset == 'all' and iou_thr == 0.5:
                        ef1 = max(f1)
                        eap = ap
                        metrics['ef1'] = ef1
                        metrics['eap'] = eap
                    if subset == 'visible' and iou_thr == 0.5:
                        eloc = self.precision_at_50()
                        eauc = auc
                        metrics['eloc'] = eloc
                        metrics['eauc'] = eauc
                metrics[f'Precision-{subset_name}'] = p
                # metrics[f'Recall-{subset_name}'] = r
                if np.isnan(f1).any():
                    metrics[f'F1-{subset_name}'] = f1
                else:
                    metrics[f'F1-{subset_name}'] = ' '.join([f'{f * 100:.1f}' for f in f1])
                metrics[f'AP-{subset_name}'] = ap
                metrics[f'AUC-{subset_name}'] = auc

        return metrics

    def gather_results(self):
        # import torch.distributed as dist
        # if not dist.is_initialized():
        return self.name_list, self.area_list, self.bb_list, self.ciou_list, self.confidence_list
        # world_size = dist.get_world_size()
        #
        # bb_list = [None for _ in range(world_size)]
        # dist.all_gather_object(bb_list, self.bb_list)
        # bb_list = [x for bb in bb_list for x in bb]
        #
        # area_list = [None for _ in range(world_size)]
        # dist.all_gather_object(area_list, self.area_list)
        # area_list = [x for area in area_list for x in area]
        #
        # ciou_list = [None for _ in range(world_size)]
        # dist.all_gather_object(ciou_list, self.ciou_list)
        # ciou_list = [x for ciou in ciou_list for x in ciou]
        #
        # confidence_list = [None for _ in range(world_size)]
        # dist.all_gather_object(confidence_list, self.confidence_list)
        # confidence_list = [x for conf in confidence_list for x in conf]
        #
        # name_list = [None for _ in range(world_size)]
        # dist.all_gather_object(name_list, self.name_list)
        # name_list = [x for name in name_list for x in name]

        # return name_list, area_list, bb_list, ciou_list, confidence_list

    def precision_at_50(self):
        ss = [i for i, bb in enumerate(self.bb_list) if bb > 0]
        return np.mean(np.array([self.ciou_list[i] for i in ss]) > 0.5)

    def precision_at_50_object(self):
        max_num_obj = max(self.bb_list)
        for num_obj in range(1, max_num_obj + 1):
            ss = [i for i, bb in enumerate(self.bb_list) if bb == num_obj]
            precision = np.mean(np.array([self.ciou_list[i] for i in ss]) > 0.5)
            print('\n' + f'num_obj:{num_obj}, precision:{precision}')

    def f1_at_50(self):
        # conf_thr = np.array(self.confidence_list).mean()
        p, r = self.calc_precision_recall(self.bb_list, self.ciou_list, self.confidence_list, self.default_conf_thr,
                                          0.5)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.

    def ap_at_50(self):
        return self.calc_ap(self.bb_list, self.ciou_list, self.confidence_list, 0.5)

    def clear(self):
        self.ciou_list = []
        self.area_list = []
        self.confidence_list = []
        self.name_list = []
        self.bb_list = []

    def update(self, bb, gt, conf, pred, pred_thr, name):
        if isinstance(conf, torch.Tensor):
            conf = conf.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        # Compute binary prediction map
        infer = np.zeros((224, 224))
        infer[pred >= pred_thr] = 1

        # Compute ciou between prediction and ground truth
        ciou = np.sum(infer * gt) / (np.sum(gt) + np.sum(infer * (gt == 0)) + 1e-12)

        # Compute ground truth size
        area = gt.sum()

        # Save
        self.confidence_list.append(conf)
        self.ciou_list.append(ciou)
        self.area_list.append(area)
        self.name_list.append(name)
        self.bb_list.append(bb)

    def evaluate_batch(self, pred, gt, label, conf, name, thr=None):
        for i in range(pred.shape[0]):
            infer = pred[i, 0].detach().cpu().numpy()
            if thr is None:
                thr = np.sort(infer.flatten())[int(infer.shape[0] * infer.shape[1] * 0.5)]

            bb = 1 if label[i] != 'non-sounding' else 0

            self.update(bb, gt[i, 0], conf[i], infer, thr, name[i])

    def finalize(self):
        metric_extend = self.finalize_stats()
        eap = metric_extend['AP-all@50']
        ef1 = metric_extend['F1-all@50']
        # eloc = metric_extend['Precision-visible@50']
        emaxf1 = max([float(num) for num in ef1.split(' ')])
        return self.metrics, {self.metrics[0]: eap*100, self.metrics[1]: emaxf1}  # , self.metrics[2]: eloc*100}
