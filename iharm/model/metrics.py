import torch
import torch.nn.functional as F


class TrainMetric(object):
    def __init__(self, pred_outputs, gt_outputs, epsilon=1e-6):
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs
        self.epsilon = epsilon
        self._last_batch_metric = 0.0
        self._epoch_metric_sum = 0.0
        self._epoch_batch_count = 0
        self._name = type(self).__name__ + '_'

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        self._last_batch_metric = self.compute(*args, **kwargs)
        self._epoch_metric_sum += self._last_batch_metric
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_metric_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_metric_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw, tag_prefix, global_step):
        sw.add_scalar(tag=tag_prefix, value=self._last_batch_metric, global_step=global_step)

    @property
    def name(self):
        return self._name


class PSNRMetric(TrainMetric):
    def __init__(self, pred_output='instances', gt_output='instances'):
        super(PSNRMetric, self).__init__((pred_output, ), (gt_output, ))
        self._name += pred_output

    def compute(self, pred, gt):
        squared_max = 255 ** 2
        mse = F.mse_loss(pred, gt)
        psnr = 10 * torch.log10(squared_max / (mse + self.epsilon))
        return psnr.item()


class DenormalizedTrainMetric(TrainMetric):
    def __init__(self, pred_outputs, gt_outputs, mean=None, std=None, color_transform=None):
        super(DenormalizedTrainMetric, self).__init__(pred_outputs, gt_outputs)
        self.mean = torch.zeros(1) if mean is None else mean
        self.std = torch.ones(1) if std is None else std
        self.restore_color = (lambda x: x) if color_transform is None else color_transform
        self.device = None

    def init_device(self, input_device):
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return self.restore_color(tensor * self.std + self.mean)

    def update(self, *args, **kwargs):
        self._last_batch_metric = self.compute(*args, **kwargs)
        self._epoch_metric_sum += self._last_batch_metric
        self._epoch_batch_count += 1


class DenormalizedPSNRMetric(DenormalizedTrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
        mean=None, std=None, color_transform=None
    ):
        super(DenormalizedPSNRMetric, self).__init__((pred_output, ), (gt_output, ), mean, std, color_transform)
        self._name += pred_output

    def compute(self, pred, gt):
        denormalized_pred = torch.clamp(self.denormalize(pred), 0, 1)
        denormalized_gt = self.denormalize(gt)
        return PSNRMetric.compute(self, denormalized_pred, denormalized_gt)

class DenormalizedPSNRMetric_FR(DenormalizedTrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
        mean=None, std=None, color_transform=None
    ):
        super(DenormalizedPSNRMetric_FR, self).__init__((pred_output, ), (gt_output, ), mean, std, color_transform)
        self._name += pred_output

    def compute(self, pred, gt):
        if torch.is_tensor(pred) and torch.is_tensor(gt):
            denormalized_pred = torch.clamp(self.denormalize(pred), 0, 1)
            denormalized_gt = self.denormalize(gt)
            return PSNRMetric.compute(self, denormalized_pred, denormalized_gt)
        else:
            psnr = 0
            for tpred, tgt in zip(pred, gt):
                psnr += self.compute(tpred.unsqueeze(0), tgt.unsqueeze(0))
            return psnr/len(pred)

class DenormalizedMSEMetric(DenormalizedTrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
        mean=None, std=None, color_transform=None
    ):
        super(DenormalizedMSEMetric, self).__init__((pred_output, ), (gt_output, ), mean, std, color_transform)
        self._name += pred_output

    def compute(self, pred, gt):
        denormalized_pred = self.denormalize(pred) * 255
        denormalized_gt = self.denormalize(gt) * 255
        return F.mse_loss(denormalized_pred, denormalized_gt).item()


class DenormalizedMSEMetric_FR(DenormalizedTrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
        mean=None, std=None, color_transform=None
    ):
        super(DenormalizedMSEMetric_FR, self).__init__((pred_output, ), (gt_output, ), mean, std, color_transform)
        self._name += pred_output

    def compute(self, pred, gt):
        if torch.is_tensor(pred) and torch.is_tensor(gt):
            denormalized_pred = self.denormalize(pred) * 255
            denormalized_gt = self.denormalize(gt) * 255
            return F.mse_loss(denormalized_pred, denormalized_gt).item()
        else:
            mse = 0
            for tpred, tgt in zip(pred, gt):
                mse += self.compute(tpred.unsqueeze(0), tgt.unsqueeze(0))
            return mse/len(pred)

class MSEMetric(TrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
    ):
        super(MSEMetric, self).__init__((pred_output, ), (gt_output, ))
        self._name += pred_output

    def compute(self, pred, gt):
        pred = pred * 255
        gt = gt * 255
        return F.mse_loss(pred, gt).item()
