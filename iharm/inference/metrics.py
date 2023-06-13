from copy import copy
import math
import torch
from pytorch_msssim import ssim

class MetricsHub:
    def __init__(self, metrics, name='', name_width=20):
        self.metrics = metrics
        self.name = name
        self.name_width = name_width

    def compute_and_add(self, *args):
        cur_result = []
        for m in self.metrics:
            if not isinstance(m, TimeMetric):
                ne = m.compute_and_add(*args)
                cur_result.append(ne)
        return cur_result
    
    def compute_and_add_none(self,):
        cur_result = []
        for m in self.metrics:
            if not isinstance(m, TimeMetric):
                ne = m.compute_and_add_none()
                cur_result.append(ne)
        return cur_result
    
    def update_time(self, time_value):
        for m in self.metrics:
            if isinstance(m, TimeMetric):
                m.update_time(time_value)

    def get_table_header(self):
        table_header = ' ' * self.name_width + '|'
        for m in self.metrics:
            table_header += f'{m.name:^{m.cwidth}}|'
        splitter = len(table_header) * '-'
        return f'{splitter}\n{table_header}\n{splitter}'

    def __add__(self, another_hub):
        merged_metrics = []
        for a, b in zip(self.metrics, another_hub.metrics):
            merged_metrics.append(a + b)
        if not merged_metrics:
            merged_metrics = copy(another_hub.metrics)

        return MetricsHub(merged_metrics, name=self.name, name_width=self.name_width)

    def __repr__(self):
        table_row = f'{self.name:<{self.name_width}}|'
        for m in self.metrics:
            table_row += f'{str(m):^{m.cwidth}}|'
        return table_row


class EvalMetric:
    def __init__(self):
        self._values_sum = 0.0
        self._count = 0
        self.cwidth = 10

    def compute_and_add(self, pred, target_image, mask):
        ne = self._compute_metric(pred, target_image, mask)
        #print(ne)
        self._values_sum += ne
        self._count += 1
        return ne
    
    def compute_and_add_none(self, ):
        self._values_sum += 0
        self._count += 1
        return 0 

    def _compute_metric(self, pred, target_image, mask):
        raise NotImplementedError

    def __add__(self, another_eval_metric):
        comb_metric = copy(self)
        comb_metric._count += another_eval_metric._count
        comb_metric._values_sum += another_eval_metric._values_sum
        return comb_metric

    @property
    def value(self):
        return self._values_sum / self._count if self._count > 0 else None

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        return f'{self.value:.5f}'

    def __len__(self):
        return self._count

class SSIM(EvalMetric):
    def _compute_metric(self, pred, target_image, mask):
        mask = mask.unsqueeze(2)
        pred = pred * mask + (target_image) * (1 - mask)
        return ssim(pred.permute(2,0,1).unsqueeze(0), target_image.permute(2,0,1).unsqueeze(0)).item()

class MSE(EvalMetric):
    def _compute_metric(self, pred, target_image, mask):
        mse = (mask.unsqueeze(2) * (pred - target_image) ** 2).mean().item()
        return mse

class SE(EvalMetric):
    def _compute_metric(self, pred, target_image, mask):
        se = (mask.unsqueeze(2) * (pred - target_image) ** 2).sum().item() / 1000000
        return se

class COS(EvalMetric):
    def _compute_metric(self, pred, target_image, mask):
        # print(torch.max(pred), torch.max(target_image))
        dist = torch.cos(pred /255.0 * 2 * math.pi - target_image /255.0 * 2 * math.pi)
        # print(dist.shape, mask.shape)
        d = (mask * dist).mean().item()
        # print(d)
        return d


class fMSE(EvalMetric):
    def _compute_metric(self, pred, target_image, mask):
        diff = mask.unsqueeze(2) * ((pred - target_image) ** 2)
        return (diff.sum() / (diff.size(2) * mask.sum() + 1e-6)).item()


class PSNR(MSE):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self._epsilon = epsilon
        self.squared_max = 255 ** 2

    def _compute_metric(self, pred, target_image, mask):
        mse = super()._compute_metric(pred, target_image, mask)
        return 10 * math.log10(self.squared_max / (mse + self._epsilon))


class N(EvalMetric):
    def _compute_metric(self, pred, target_image, mask):
        return 0

    @property
    def value(self):
        return self._count

    def __repr__(self):
        return str(self.value)


class TimeMetric(EvalMetric):
    def update_time(self, time_value):
        self._values_sum += time_value
        self._count += 1


class AvgPredictTime(TimeMetric):
    def __init__(self):
        super().__init__()
        self.cwidth = 14

    @property
    def name(self):
        return 'AvgTime, ms'

    def __repr__(self):
        return f'{1000 * self.value:.1f}'
