#!/usr/bin/python
# -*- coding: UTF-8 -*-


import zmq
import random
import sys
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://localhost:10020")

topic = 'Test'
while True:
    publisher_id = random.randrange(0,9999)
    messagedata = "server#%s" % publisher_id
    print ("%s %s" % (topic, messagedata))
    socket.send_string( topic+':' + messagedata)
    time.sleep(1)
