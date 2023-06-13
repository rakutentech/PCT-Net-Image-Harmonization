import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import copy

class BaseHDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 input_transform=None,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 epoch_len=-1):
        super(BaseHDataset, self).__init__()
        self.epoch_len = epoch_len
        self.input_transform = input_transform
        self.augmentator = augmentator
        self.keep_background_prob = keep_background_prob
        self.with_image_info = with_image_info

        if input_transform is None:
            input_transform = lambda x: x

        self.input_transform = input_transform
        self.dataset_samples = None

    def __getitem__(self, index):
        if self.epoch_len > 0:
            index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        image = self.input_transform(sample['image'])
        target_image = self.input_transform(sample['target_image'])
        obj_mask = sample['object_mask'].astype(np.float32)

        output = {
            'images': image,
            'masks': obj_mask[np.newaxis, ...].astype(np.float32),
            'target_images': target_image
        }

        if self.with_image_info and 'image_id' in sample:
            output['image_info'] = sample['image_id']
        return output

    def check_sample_types(self, sample):
        assert sample['image'].dtype == 'uint8'
        if 'target_image' in sample:
            assert sample['target_image'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.augmentator is None:
            return sample

        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augmentator.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augmentator(image=sample['image'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['object_mask'].sum() > 1.0

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return len(self.dataset_samples)

class BaseHDatasetUpsample(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator_1=None,
                 augmentator_2=None,
                 input_transform=None,
                 keep_background_prob=0.0,
                 with_image_info=True,
                 use_hr=True,
                 color_jitter=None,
                 epoch_len=-1):
        super(BaseHDatasetUpsample, self).__init__()
        self.epoch_len = epoch_len
        self.input_transform = input_transform
        self.augmentator_1 = augmentator_1
        self.augmentator_2 = augmentator_2
        self.keep_background_prob = keep_background_prob
        self.with_image_info = with_image_info
        self.use_hr = use_hr
        self.color_jitter = color_jitter

        if input_transform is None:
            input_transform = lambda x: x

        self.input_transform = input_transform
        self.dataset_samples = None


    def extract_bboxes(self, m):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([4], dtype=np.int32)
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        #print("np.any(m, axis=0)",np.any(m, axis=0))
        #print("p.where(np.any(m, axis=0))",np.where(np.any(m, axis=0)))
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)

    def __getitem__(self, index):
        if self.epoch_len > 0:
            index = random.randrange(0, len(self.dataset_samples))

        sample_1 = self.get_sample(index)
        self.check_sample_types(sample_1)
        if self.use_hr:
            sample_fullres = self.augment_sample(sample_1, self.augmentator_1)
            old_fullres = sample_fullres['image'].copy()
            
            if self.color_jitter:
                sample_jitter = self.color_jitter(image=sample_fullres['image'])
                tmp_mask = sample_fullres['object_mask'][..., np.newaxis].astype(np.uint8)
                sample_fullres['image'] = old_fullres * (1 - tmp_mask) + \
                                            sample_jitter['image'] * tmp_mask

            sample_lowres = self.augment_sample(sample_fullres, self.augmentator_2)
            image = self.input_transform(sample_lowres['image'])
            target_image = self.input_transform(sample_lowres['target_image'])

            image_fullres = self.input_transform(sample_fullres['image'])
            target_image_fullres = self.input_transform(sample_fullres['target_image'])

            obj_mask = sample_lowres['object_mask'].astype(np.float32)
            obj_mask_fullres = sample_fullres['object_mask'].astype(np.float32)
            
            output = {
                'images': image,
                'masks': obj_mask[np.newaxis, ...],
                'target_images': target_image,
                'images_fullres' : image_fullres,
                'masks_fullres' : obj_mask_fullres[np.newaxis, ...],
                'target_images_fullres' : target_image_fullres,
            }
        else:
            sample_lowres = self.augment_sample(sample_1, self.augmentator_2)
            if self.color_jitter:
                sample_jitter = self.augment_sample(sample_lowres, self.color_jitter)
                sample_lowres['image'] = sample_lowres['image'] * (1 - sample_lowres['object_mask'][..., np.newaxis]) + \
                                            sample_jitter['image'] * sample_lowres['object_mask'][..., np.newaxis]

            image = self.input_transform(sample_lowres['image'])
            target_image = self.input_transform(sample_lowres['target_image'])
            obj_mask = sample_lowres['object_mask'].astype(np.float32)

            output = {
                'images': image,
                'masks': obj_mask[np.newaxis, ...],
                'target_images': target_image,
                'images_fullres' : image,
                'masks_fullres' : obj_mask[np.newaxis, ...]
            }

        if self.with_image_info and 'image_id' in sample_1:
            output['image_info'] = sample_1['image_id']
        return output

    def check_sample_types(self, sample):
        assert sample['image'].dtype == 'uint8'
        if 'target_image' in sample:
            assert sample['target_image'].dtype == 'uint8'

    def augment_sample(self, sample, augmentator):
        if augmentator is None:
            return sample

        additional_targets = {target_name: sample[target_name]
                              for target_name in augmentator.additional_targets.keys()}

        valid_augmentation = False
        cnt = 0
        while not valid_augmentation:
            aug_output = augmentator(image=sample['image'], mask=sample['object_mask'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        new_sample = dict()
        for target_name, transformed_target in aug_output.items():
            new_sample[target_name] = transformed_target

        return new_sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['object_mask'].sum() > 1.0

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return len(self.dataset_samples)

