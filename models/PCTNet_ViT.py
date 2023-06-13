import torch
from torchvision import transforms
from functools import partial
from easydict import EasyDict as edict
from albumentations import Resize
from kornia.color import RgbToHsv, HsvToRgb
from torch.nn import init

from iharm.data.compose import ComposeDatasetUpsample
from iharm.data.hdataset import HDatasetUpsample
from iharm.data.transforms import HCompose, RandomCropNoResize

from iharm.model.base.pct_net import PCTNet
from iharm.model.losses import MaskWeightedMSE, ParamSmoothness, SCS_CR_loss 
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric
from iharm.engine.PCTNet_trainer import PCT_Trainer
from iharm.mconfigs import BMCONFIGS
from iharm.utils.log import logger

def main(cfg):
    model, model_cfg, ccfg = init_model(cfg)
    train(model, cfg, model_cfg, ccfg, start_epoch=cfg.start_epoch)


def init_func(m, init_gain=0.02):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (768, 1024)
    model_cfg.input_normalization = {
        'mean': [0, 0, 0],
        'std': [1, 1, 1]
    }

    ccfg = BMCONFIGS['ViT_pct']
    ccfg['params']['input_normalization'] = model_cfg.input_normalization   
    model = PCTNet(**ccfg['params'])
    model.apply(init_func)                                                  # apply the initialization function <init_func>

    input_transform = [transforms.ToTensor()]
    if ccfg['data']['color_space'] == 'HSV':
        input_transform.append(RgbToHsv())
        input_transform.append(transforms.Normalize([0, 0, 0], [6.283, 1, 1]))
    input_transform.append(transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']))
    model_cfg.input_transform = transforms.Compose(input_transform)

    return model, model_cfg, ccfg

def train(model, cfg, model_cfg, ccfg, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeightedMSE(min_area=1000,  pred_name='images_fullres', 
            gt_image_name='target_images_fullres', gt_mask_name='masks_fullres')
    loss_cfg.pixel_loss_weight = 1.0

    loss_cfg.contrastive_loss = SCS_CR_loss()
    loss_cfg.contrastive_loss_weight = 0.01

    loss_cfg.smooth_loss = ParamSmoothness()
    loss_cfg.smooth_loss_weight = 0.1
    
    blur = False
    use_hr = True
    
    num_epochs = 100
    low_res_size = (256, 256)

    train_augmentator_1 = HCompose([
        RandomCropNoResize(ratio=0.7)
    ])
    train_augmentator_2 = HCompose([
        Resize(*low_res_size)
    ])

    val_augmentator_1 = None
    val_augmentator_2 = HCompose([
        Resize(*low_res_size)
    ])

    blur = False
    trainset = ComposeDatasetUpsample(
        [
            HDatasetUpsample(cfg.HFLICKR_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HDAY2NIGHT_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HCOCO_PATH, split='train', blur_target=blur),
            # HDatasetUpsample(cfg.HADOBE5K_PATH, split='train', blur_target=blur),
        ],
        augmentator_1=train_augmentator_1,
        augmentator_2=train_augmentator_2,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05,
        use_hr=True
    )

    valset = ComposeDatasetUpsample(
        [
            HDatasetUpsample(cfg.HFLICKR_PATH, split='test', blur_target=blur, mini_val=False),
            HDatasetUpsample(cfg.HDAY2NIGHT_PATH, split='test', blur_target=blur, mini_val=False),
            HDatasetUpsample(cfg.HCOCO_PATH, split='test', blur_target=blur, mini_val=False),
            # HDatasetUpsample(cfg.HADOBE5K_PATH, split='test', blur_target=blur)
        ],
        augmentator_1=val_augmentator_1,
        augmentator_2=val_augmentator_2,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1,
        use_hr=True
    )

    if len(cfg.gpu_ids) > 1:
        optimizer_params = {
            'lr': 1e-4 * float(cfg.batch_size) / 16.0 * len(cfg.gpu_ids) / 2.0,
            'betas': (0.9, 0.999), 'eps': 1e-8
        }
    else:
        optimizer_params = {
            'lr': 1e-4 * float(cfg.batch_size) / 16.0,
            'betas': (0.9, 0.999), 'eps': 1e-8
        }

    if cfg.local_rank == 0:
        print(optimizer_params)

    scheduler1 = partial(torch.optim.lr_scheduler.ConstantLR, factor=1)
    scheduler2 = partial(torch.optim.lr_scheduler.LinearLR, start_factor=1, end_factor=0, total_iters=50)
    lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1(optimizer=optimizer), scheduler2(optimizer=optimizer)], 
                                                                            milestones=[50])

    if ccfg['data']['color_space'] == 'HSV':
        color_transform = transforms.Compose([HsvToRgb(), transforms.Normalize([0, 0, 0], [-6.283, 1, 1])])
    else:
        color_transform = None

    def collate_fn_FR(batch): 
        # Initializae dictionary
        keys = ['images', 'masks', 'target_images', 'images_fullres', 'masks_fullres', 'target_images_fullres', 'image_info']
        bdict = {}
        for k in keys:
            bdict[k] = []
        
        # Create batched dictionary
        for elem in batch:
            for key in keys:
                if key in ['masks', 'masks_fullres']:           
                    elem[key] = torch.tensor(elem[key])
                bdict[key].append(elem[key])


        bdict['images'] = torch.stack(bdict['images'])
        bdict['target_images'] = torch.stack(bdict['target_images'])
        bdict['masks'] = torch.stack(bdict['masks'])

        return bdict

    trainer = PCT_Trainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[
            DenormalizedPSNRMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                color_transform=color_transform,
            ),
            DenormalizedMSEMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                color_transform=color_transform,
            ),
        ],
        checkpoint_interval=1,
        image_dump_interval=100,
        color_space=ccfg['data']['color_space'],
        collate_fn = collate_fn_FR,
        random_swap=0,
        random_augment=False
    )

    if cfg.local_rank == 0:
        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
