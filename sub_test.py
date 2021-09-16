import sys
import zmq
import json

# Socket to talk to server
context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://210.107.197.247:10010")
subscriber.setsockopt(zmq.CONFLATE, 1) 
topic = "result"
subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
while(True):
        string = subscriber.recv()
        topic, data = string.split(b' ',1)
        parsed = json.loads(data)
        print(json.dumps(parsed, indent=4, sort_keys=True))
