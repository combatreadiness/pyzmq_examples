#!/usr/bin/python
# -*- coding: UTF-8 -*-


import sys
import zmq

# Socket to talk to server
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://127.0.0.1:10010")
topic = "Test"
subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
while(True):
        string = subscriber.recv()
        print (string)
