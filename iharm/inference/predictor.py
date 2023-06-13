import torch
from torchvision.transforms import Normalize

from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor, RGB2HSV

class Predictor(object):
    def __init__(self, net, device, with_flip=False,
                 mean=(.485, .456, .406), std=(.229, .224, .225), hsv=False, use_attn=True):
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()
        
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)

        self.transforms = [ToTensor(self.device)]
        if hsv:
            self.transforms.append(RGB2HSV())
            self.transforms.append(NormalizeTensor(torch.zeros(3), torch.tensor([6.283, 1, 1]), self.device))
        self.transforms.append(NormalizeTensor(mean, std, self.device))

        self.norm = Normalize(mean, std)
        self.unnorm = Normalize(mean=-mean/std, std=1/std)

        if with_flip:
            self.transforms.append(AddFlippedTensor())
        self.avgs = []

        self.use_attn = use_attn

    def predict(self, image, image_highres, mask, mask_highres, return_numpy=True):

        with torch.no_grad():
            for transform in self.transforms:
                image, mask = transform.transform(image, mask)
            input_mask = mask
            for transform in self.transforms:
                image_highres, mask_highres = transform.transform(image_highres, mask_highres)
            
            output = self.net(image.float(), image_highres.float(), input_mask.float(), mask_highres.float())
            predicted_image = output['images']
            output_fullres = output['images_fullres'].unsqueeze(0)

            for transform in reversed(self.transforms):
                predicted_image = transform.inv_transform(predicted_image)

            for transform in reversed(self.transforms):
                output_fullres = transform.inv_transform(output_fullres)

            predicted_image = torch.clamp(predicted_image, 0, 255)
            output_fullres = torch.clamp(output_fullres, 0, 255)

        if return_numpy:
            return predicted_image.cpu().numpy(), output_fullres.cpu().numpy() 
        else:
            return predicted_image, output_fullres
