import os
import sys
import cv2
import numpy as np
import skimage.exposure
import torch
from torchvision import transforms

from model import HDRPointwiseNN
from utils import load_image, resize, load_params
import matplotlib.pyplot as plt

def test(ckpt, args={}):
    state_dict = torch.load(ckpt)
    state_dict, params = load_params(state_dict)
    params.update(args)

    device = torch.device("cuda")
    tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    if not os.path.exists(params['output']):
        os.makedirs(params['output'])
    
    input_images = os.listdir(params['input'])

    low = tensor(resize(load_image(params['test_image']),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
    full = tensor(load_image(params['test_image']).astype(np.float32)).repeat(1,1,1,1)/255
    
    low = low.to(device)
    full = full.to(device)
    with torch.no_grad():
        model = HDRPointwiseNN(params=params)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        for image in input_images:
            low = tensor(resize(load_image(os.path.join(params['input'], image)),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
            full = tensor(load_image(os.path.join(params['input'], image)).astype(np.float32)).repeat(1,1,1,1)/255
            img = model(low, full)        
            img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['test_out'], image), img[...,::-1])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--checkpoint', type=str, help='model state path')
    parser.add_argument('--input', type=str, dest="test_image", help='image path')
    parser.add_argument('--output', type=str, dest="test_out", help='output image path')

    args = vars(parser.parse_args())

    test(args['checkpoint'], args)