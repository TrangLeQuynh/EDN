import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from saleval import SalEval
from models import model as net
from tqdm import tqdm
import time
from tools.utils import check_size
import math
import os

@torch.no_grad()
def run_inference(args, model, img_path, save_dir):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    eval = SalEval()

    image = cv2.imread(img_path)

    # resize the image to 1024x512x3 as in previous papers
    img = cv2.resize(image, (args.width, args.height))
    img = img.astype(np.float32) / 255.
    img -= mean
    img /= std

    img = img[:,:, ::-1].copy()
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    img = Variable(img)

    if args.gpu:
        img = img.cuda()
    begin = time.time()
    img_out = model(img)
    img_out = img_out[:, 0, :, :].unsqueeze(dim=0)
    end = time.time()
    print("Time: %.4f ms" % ((end - begin) * 1000))

    img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)

    sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
    cv2.imwrite(osp.join(save_dir, os.path.basename(img_path)), sal_map)


def build_model(args):
    model = net.EDN(arch=args.arch)
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(args.pretrained)
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        new_keys.append(key.replace('module.', ''))
        new_values.append(value)
        new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict, strict=False)

    if args.gpu:
        model = model.cuda()
    # set to evaluation mode
    model.eval()

    return model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
    parser.add_argument('--gpu', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', type=str, help='Pretrained model', required=True)
    parser.add_argument("--source", '-s', type=str, required=True)

    args = parser.parse_args()

    if 'Lite' in args.pretrained:
        args.arch = 'mobilenetv2'
    elif 'VGG16' in args.pretrained:
        args.arch = 'vgg16'
    elif 'R50' in args.pretrained:
        args.arch = 'resnet50'
    elif 'P2T-S' in args.pretrained:
        args.arch = 'p2t_small'
    else:
        raise NotImplementedError("recognized unknown backbone given the model_path")

    if 'LiteEX' in args.pretrained:
        # EDN-LiteEX
        args.width = 224
        args.height = 224
    
    return args

if __name__ == '__main__':
    args = parse_args()
    print("Device: {}".format("gpu" if args.gpu else "cpu"))
    model = build_model(args=args)
    if not osp.isdir(args.savedir):
        os.makedirs(args.savedir)
    run_inference(
        args=args,
        model=model,
        img_path=args.source,
        save_dir=args.savedir
    )
