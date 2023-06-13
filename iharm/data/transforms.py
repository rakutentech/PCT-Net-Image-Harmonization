from albumentations import Compose, LongestMaxSize, DualTransform, ImageOnlyTransform
import albumentations.augmentations.functional as F
import albumentations.augmentations.crops.functional as Fc
from albumentations.core.transforms_interface import DualTransform
import albumentations
import cv2
from kornia.color import hsv_to_rgb
import torch
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

import random

class HCompose(Compose):
    def __init__(self, transforms, *args, additional_targets=None, no_nearest_for_masks=True, **kwargs):
        if additional_targets is None:
            additional_targets = {
                'target_image': 'image',
                'object_mask': 'mask'
            }
        self.additional_targets = additional_targets
        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)
        if no_nearest_for_masks:
            for t in transforms:
                if isinstance(t, DualTransform):
                    t._additional_targets['object_mask'] = 'image'


class RGB_to_HSV(ImageOnlyTransform):

    def __init__(self, always_apply: bool = True, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        else:
            return img 

    def invert(self, img: torch.tensor):
        return hsv_to_rgb(img)

class RandomCropNoResize(DualTransform):

    def __init__(self, ratio: float, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        '''
        ratio: ratio applied to scale height and width of crop, randomly chosen from (ratio, 1)
        '''
        self.ratio = ratio

    def get_params(self):
        scale_height = random.uniform(self.ratio, 1)
        scale_width = random.uniform(self.ratio, 1)
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "scale_height": scale_height,
            "scale_width": scale_width,
        }

    def apply(self, img, scale_height=0, scale_width=0, h_start=0, w_start=0, **params):
        return Fc.random_crop(img, int(scale_height*img.shape[0]), int(scale_width*img.shape[1]), h_start, w_start)

    def get_transform_init_args_names(self):
        return "ratio", "height", "width"

class LongestMaxSizeIfLarger(LongestMaxSize):
    """
    Rescale an image so that maximum side is less or equal to max_size, keeping the aspect ratio of the initial image.
    If image sides are smaller than the given max_size, no rescaling is applied.

    Args:
        max_size (int): maximum size of smallest side of the image after the transformation.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        if max(img.shape[:2]) < self.max_size:
            return img
        return albumentations.geometric.functional.longest_max_size(img, max_size=self.max_size, interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]

        scale = self.max_size / max([height, width])
        if scale > 1.0:
            return keypoint
        return F.keypoint_scale(keypoint, scale, scale)
