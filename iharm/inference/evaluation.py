from time import time
from tqdm import trange
import os
import cv2
import numpy as np
import torch
import pandas as pd

def to_image(x):
    return x[0].permute((1,2,0)).cpu().numpy() * 255

def to_eval(x):
    return x[0].permute((1,2,0))         

def evaluate_dataset(dataset, predictor, metrics_hub_lowres, metrics_hub_fullres, visdir=None, csv_dir=None):

    metric_data = []
    for sample_i in trange(len(dataset), desc=f'Testing on {metrics_hub_lowres.name}'):

        bdata = dataset.get_sample(sample_i)
        imname = dataset.dataset_samples[sample_i]

        imname = imname.replace('.jpg', '')
        
        raw_image = bdata['image']
        raw_target = bdata['target_image']
        raw_mask = bdata['object_mask']
        sample = dataset.augment_sample(bdata, dataset.augmentator_2)
        sample_highres = dataset.augment_sample(bdata, dataset.augmentator_1)
        image_lowres = sample['image']
        sample_mask = sample['object_mask']
        sample_mask_highres = sample_highres['object_mask']
        
        predict_start = time()
        pred, pred_fullres = predictor.predict(image_lowres, raw_image, sample_mask, sample_mask_highres, return_numpy=False)
        pred = torch.as_tensor(pred, dtype=torch.float32).to(predictor.device)
        pred_fullres = torch.as_tensor(pred_fullres, dtype=torch.float32).to(predictor.device)
        
        if predictor.device.type != 'cpu':
            torch.cuda.synchronize()
        metrics_hub_lowres.update_time(time() - predict_start)
        metrics_hub_fullres.update_time(time() - predict_start)

        target_image = torch.as_tensor(sample['target_image'], dtype=torch.float32).to(predictor.device)
        sample_mask = torch.as_tensor(sample_mask, dtype=torch.float32).to(predictor.device)

        raw_target = torch.as_tensor(raw_target, dtype=torch.float32).to(predictor.device)
        raw_mask = torch.as_tensor(raw_mask, dtype=torch.float32).to(predictor.device)
        raw_image = torch.as_tensor(raw_image, dtype=torch.float32).to(predictor.device)

        with torch.no_grad():
            metrics_hub_lowres.compute_and_add(pred, target_image, sample_mask)
            fullres_result = metrics_hub_fullres.compute_and_add(pred_fullres, raw_target, raw_mask)

        mse = fullres_result[1]
        psnr = fullres_result[2]
        fmse = fullres_result[3]
        se = fullres_result[4]
        ssim = fullres_result[5]
        height, width = raw_mask.shape
        mask_area = raw_mask.sum().item()
        metric_data.append([imname, mse, psnr, fmse, se, ssim, height, width, mask_area])

        if visdir and sample_i % 1 == 0:

            raw_mask = raw_mask.cpu().numpy()
            raw_mask = np.stack([raw_mask]*3, axis=2) * 255
            pred_fullres = pred_fullres.cpu().numpy()
            
            # Fullres
            cv2.imwrite(os.path.join(visdir, f'{imname}_harmonized.jpg'), pred_fullres[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join(visdir, f'{imname}_real.jpg'), bdata['target_image'][:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join(visdir, f'{imname}_mask.jpg'), raw_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join(visdir, f'{imname}_comp.jpg'), raw_image.cpu().numpy()[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
            # Lowres
            # pred = pred.cpu().numpy()
            # target_image = target_image.cpu().numpy()
            # sample_mask = sample_mask.cpu().numpy()
            # sample_mask = np.stack([sample_mask]*3, axis=2) * 255
            # image_lowres = image_lowres
            # cv2.imwrite(os.path.join(visdir, f'{imname}_harmonized.jpg'), pred[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(os.path.join(visdir, f'{imname}_real.jpg'), target_image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(os.path.join(visdir, f'{imname}_mask.jpg'), sample_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(os.path.join(visdir, f'{imname}_comp.jpg'), image_lowres[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
    df = pd.DataFrame(metric_data, columns=['Name', 'MSE', 'PSNR', 'fMSE', 'SE', 'SSIM', 'height', 'width', 'mask_area'])
    df.to_csv(str(csv_dir).replace(".log", ".csv"), mode='a', header=not os.path.exists(str(csv_dir).replace('.log', '.csv')), index=False)
