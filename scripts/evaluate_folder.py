import argparse
import os
import sys
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model
from iharm.mconfigs import ALL_MCONFIGS

parser = argparse.ArgumentParser()
parser.add_argument('src', help='Source directory')
parser.add_argument('model_type', help='model type')
parser.add_argument('--weights', help='path to the weights')
parser.add_argument('--gpu', type=str, default='cpu', help='ID of used GPU.')

args = parser.parse_args()

# Load model 
model = load_model(args.model_type, args.weights, verbose=False)

device = torch.device(args.gpu)
use_attn = ALL_MCONFIGS[args.model_type]['params']['use_attn']
normalization = ALL_MCONFIGS[args.model_type]['params']['input_normalization']
predictor = Predictor(model, device, use_attn=use_attn, mean=normalization['mean'], std=normalization['std'])

# Get data
file_list = sorted(os.listdir(args.src))
imgs = [img for img in file_list if img.split('_')[-1] in ['mask.jpg', 'mask.png']]

for img in tqdm(imgs):  

    # Load images      
    comp = cv2.imread(os.path.join(args.src, img[:-9] + '.jpg'))
    comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(args.src, img), cv2.IMREAD_GRAYSCALE) / 255
    comp_lr = cv2.resize(comp, (256, 256))
    mask_lr = cv2.resize(mask, (256, 256))
    
    # Inference
    pred_lr, pred_img = predictor.predict(comp_lr, comp, mask_lr, mask)

    # Save Image
    cv2.imwrite(os.path.join(args.src, img[:-9] + f'_harm.jpg'), pred_img[:,:,::-1])
