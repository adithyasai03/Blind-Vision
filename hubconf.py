

dependencies = ['torch', 'yaml']
import os

import torch

from models.yolo import Model
from utils.google_utils import attempt_download


def create(name, pretrained, channels, classes):
    config = os.path.join(os.path.dirname(__file__), 'models', '%s.yaml' % name)  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            ckpt = '%s.pt' % name  
            attempt_download(ckpt)  
            state_dict = torch.load(ckpt, map_location=torch.device('cpu'))['model'].float().state_dict()  # to FP32
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
            model.load_state_dict(state_dict, strict=False)  # load
        return model

    except Exception as e:
        
        s = 'Cache maybe be out of date, deleting cache and retrying may solve this. See %s for help.' % help_url
        raise Exception(s) from e


def yolov5s(pretrained=False, channels=3, classes=80):
   
    return create('yolov5s', pretrained, channels, classes)


def yolov5m(pretrained=False, channels=3, classes=80):
    
    return create('yolov5m', pretrained, channels, classes)


def yolov5l(pretrained=False, channels=3, classes=80):
    
    return create('yolov5l', pretrained, channels, classes)


def yolov5x(pretrained=False, channels=3, classes=80):
   
    return create('yolov5x', pretrained, channels, classes)
