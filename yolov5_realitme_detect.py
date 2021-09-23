import argparse
import sys
import cv2
import numpy as np
import zmq
import json
import torch


class zmqHandler():


    def __init__(self, host, port, topic):

        self.host = host
        self.port = port
        self.topic = topic
       #self.socket

    def subscirbeInit(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect ("tcp://%s:%s" %(self.host,self.port))
        self.socket.setsockopt(zmq.CONFLATE, 1) # subscribe only latest data
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def subscribe(self):
        topic = self.socket.recv()
        framedata = self.socket.recv()
        return framedata

    def publishInit(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.connect ("tcp://%s:%s" %(self.host,self.port))
        
    
    def publish(self, messagedata):
        self.socket.send_string("%s %s" %(self.topic,messagedata))
                

    

def detect(option_list):

    #zeromq init
    zmqSubscriber = zmqHandler(option_list.subHost, option_list.subPort, option_list.subTopic)
    zmqSubscriber.subscirbeInit()
    zmqPublisher = zmqHandler(option_list.pubHost, option_list.pubPort, option_list.pubTopic)
    zmqPublisher.publishInit()

    #model init
    model = torch.hub.load('.', 'yolov5s', pretrained=True, source='local')

# Inference
    while True:
        
        framedata = zmqSubscriber.subscribe()
        
        try:
                
            framedata = np.frombuffer(framedata, dtype='uint8')
            framedata = cv2.imdecode(framedata, cv2.IMREAD_COLOR)
            framedata = cv2.cvtColor(framedata, cv2.COLOR_BGR2RGB)
            results = model(framedata)
            
            zmqPublisher.publish((results.pandas().xyxy[0].to_json(orient='records')))
        

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pubHost',  nargs=1, type=str, help='Forwarder IP address for publish', default='210.107.197.247', dest='pubHost')
    parser.add_argument('--pubPort',  nargs=1, type=str, help='Forwarder port number for publish', default='10020', dest='pubPort')
    parser.add_argument('--pubTopic', nargs=1, type=str, help='Publish Topic', default='result', dest='pubTopic')
    parser.add_argument('--subHost',  nargs=1, type=str, help='Forwarder IP address for subscribe', default='210.107.197.247', dest='subHost')
    parser.add_argument('--subPort',  nargs=1, type=str, help='Forwarder port number for subscribe', default='10010', dest='subPort')
    parser.add_argument('--subTopic', nargs=1, type=str, help='subscribe Topic', default='test', dest='subTopic')

    option_list = parser.parse_args()
    return option_list




if __name__ == '__main__':
    option_list = get_arguments()   
    with torch.no_grad():
            detect(option_list)
