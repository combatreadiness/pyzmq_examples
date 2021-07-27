import sys

import zmq
from zmq import ZMQError

import redis

from io import BlockingIOError, BytesIO
import io
import numpy as np

import cv2

import msgpack
import msgpack_numpy as m
m.patch()

import matplotlib.pyplot as plt

import threading
from threading import Thread

from IPython.display import display, clear_output

class ZmqToRedisHandler(Thread):
    def __init__(self, zmqHost, zmqPort, zmqTopic, redisHost, redisPort, redisNo ):
        self.zmqHost=zmqHost
        self.zmqPort=zmqPort
        self.zmqTopic=zmqTopic
        
        self.redisHost=redisHost
        self.redisPort=redisPort
        self.redisNo=redisNo
                
        self.context=None
        self.socket=None
        
        self.redis = redis.Redis(host=self.redisHost, port=self.redisPort, db=self.redisNo)
        
        Thread.__init__(self)
        
        
    def connect(self, context, socket, zmq_mode, port):
        '''Connects to a forwarder proxy and returns the connection'''
        if context is None and socket is None:
            print(f">>>attemping proxy connection to {zmq_mode} at {port}")
            try:
                context = zmq.Context()
                socket = context.socket(zmq_mode)
                socket.connect(f"tcp://{self.zmqHost}:{port}")
                print(f">>>connected to proxy socket finished: with {zmq_mode} at {self.zmqHost}:{port}")
                return context, socket
            except zmq.ZMQError as ze:
                print(f">>>error connecting to proxy: with {zmq_mode} at {port}: {ze}")
        
        
    def subscribe(self, socket, topic):
        print(f">>>subscribing to topic: {topic} @ socket: {socket}...")
        socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        
        
    def receive(self, context, socket, flags=0):
        print(">>>begin receiving images...")
        
        #received ZMQ Frame will be ['topic', 'data']
        topic=None
        data=None
        
        while (True):
            #receive image data and buffer them images.
            try:
                #device_id = socket.recv(flags)
                topic = socket.recv()
                msgtype = socket.recv()
                data = socket.recv()
                
                if data is not None and data != '':
                    framedata = np.frombuffer(data, dtype='uint8')
                    serialized = m.packb(framedata)
                    self.redis.set('cam',serialized)
                    #store the serialized ndarray into redis
                    #deserialized= self.redis.get('cam')
                    #framedata= m.unpackb( deserialized )
                    #framedata = cv2.imdecode(framedata, cv2.IMREAD_COLOR)
                    #framedata = cv2.cvtColor(framedata, cv2.COLOR_BGR2RGB)
                    #plt.imshow(framedata)
                    #plt.show()
                else:
                    self.redis.set('cam','')                    
                                                               
            except BlockingIOError as ioe:
                print(f'>>>io module has a problem doing io due to blocking!: {ioe}')
            except ZMQError as ze:
                print(f'>>>Error encountered in ZMQ recv: {ze}')
            except Exception as e:
                print(f'>>>Error receiving images from ZMQ subscription(s): {e}')
     
    def run(self):        
        #create ZMQ connection
        context, socket = self.connect(self.context, self.socket, zmq.SUB, self.zmqPort)
        if context is not None and socket is not None:
            #set topic for subscription 
            self.subscribe(socket, self.zmqTopic)
            #begin receiving data
            self.receive(context, socket)
            print(f">>>video subscription to {self.zmqTopic} terminated...")
        else:
            print(f">>>proxy connection problem: context and socket empty...")
           
          
def main():
    thr0 = ZmqToRedisHandler(zmqHost='210.107.197.247',zmqPort=10010, zmqTopic='Test', redisHost='210.107.197.247', redisPort=6379, redisNo=1)
    thr0.start()
    thr0.join()
            
      
if __name__ == '__main__':
    main()
    print(f">>>Image_Subscriber_Thread Terminated...")
