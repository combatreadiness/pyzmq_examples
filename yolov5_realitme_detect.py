import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.torch_utils import select_device

import zmq

import threading
from threading import Thread



subHost = "210.107.197.247"
pubHost = "210.107.197.247"
subPort = "10010"
pubPort = "10020"
# Socket to talk to server



class yoloDetector():

    #yolo parameters
    SOURCE = 'data/images/DJI_0004.jpg'
    WEIGHTS = 'yolov5s.pt'
    IMG_SIZE = 640
    DEVICE = ''
    AUGMENT = False
    CONF_THRES = 0.25
    IOU_THRES = 0.45
    CLASSES = None
    AGNOSTIC_NMS = False

    def __init__(self):
        source, weights, imgsz = self.SOURCE, self.WEIGHTS, self.IMG_SIZE
        device = select_device(self.DEVICE)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        print('device:', device)

        
    # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

    # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        if device.type != 'cpu':
           model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        

class zmqHandler(self):

    subHost = "210.107.197.247"
    pubHost = "210.107.197.247"
    subPort = "10010"
    pubPort = "10020"

    def __init__(self):


def detect():

    detector = yoloDetector()



    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    print ("Collecting updates from server...")
    socket.connect ("tcp:/%s:%s" %(subHost,subPort))
    topicfilter = "test"
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

    for update_nbr in range(1000):
        topic = socket.recv()
        msgtype = socket.recv()
        framedata = socket.recv()
        framedata = np.frombuffer(framedata, dtype='uint8')
        framedata = cv2.imdecode(framedata, cv2.IMREAD_COLOR)
        framedata = cv2.cvtColor(framedata, cv2.COLOR_BGR2RGB)
        
        # Load image
        img0 = framedata  # BGR
        assert img0 is not None, 'Image Not Found ' + source

        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t0 = time.time()
        pred = model(img, augment=AUGMENT)[0]
        #print('pred shape:', pred.shape)

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

        # Process detections
        det = pred[0]
        # print('det shape:', det.shape)

        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string

        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
    #            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

            #print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

        # Stream results
        print(s)
        #cv2.imshow(source, img0)
        #cv2.waitKey(0)  # 1 millisecond





def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pubHost', '-p', nargs=1, type=str, help='ZeroMQ Forwarder Host IP address', default='210.107.197.247', dest='zmqHost')
    parser.add_argument('--pubPort', '-', nargs=1, type=int, help='ZeroMQ Forwarder port number', default=[10010], dest='zmqPort')
    parser.add_argument('--pubTopic', '-t', nargs=1, type=str, help='ZeroMQ Topic', default='test', dest='zmqTopic')
    parser.add_argument('--subHost', '-r', nargs=1, type=str, help='Redis IP address', default='210.107.197.247', dest='redisHost')
    parser.add_argument('--subPort', '-o', nargs=1, type=int, help='Redis port number', default=[6379], dest='redisPort')
    parser.add_argument('--subTopic', '-n', nargs=1, type=int, help='Redis db number', default=[1], dest='redisNumber')

    option_list = parser.parse_args()
    return option_list




if __name__ == '__main__':
    check_requirements(exclude=('pycocotools', 'thop'))
    option_list = get_arguments()
    with torch.no_grad():
            detect()
