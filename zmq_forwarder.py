"""
ZMQ Forwarder service by Seungwoon Lee
"""

import sys
import argparse
import zmq
import threading
from threading import Thread


class Forwarder(Thread):
    def __init__(self,front_port, back_port):
        self.front_port = front_port
        self.back_port = back_port
        threading.Thread.__init__(self)

    def create_context(self):
        '''creates a ZMQ context for the subsequent XPUB/XSUB architecture to be built here in the class'''
        self.context = zmq.Context()
                
    def bind_front(self):
        '''creates an XPUB socket and binds it to a given port'''
        self.frontend = self.context.socket(zmq.XPUB)
        try:
            self.frontend.bind(f"tcp://*:{self.front_port}")
            print(f'>>>frontend bound at {self.front_port}')
        except zmq.ZMQError as ze:
            print(f">>>Error binding frontend zmq socket {self.front_port}: {ze}")
        except Exception as e:
            print(f">>>Error attempting to bind {self.front_port}: {e}")
    
    def bind_back(self):
        '''creates an XSUB socket and binds it to a given port'''
        self.backend = self.context.socket(zmq.XSUB)
        
        try:
            self.backend.bind(f"tcp://*:{self.back_port}")
            print(f'>>>backend bound at {self.back_port}')
        except zmq.ZMQError as ze:
            print(f">>>Error binding backend zmq socket {self.back_port}: {ze}")
        except Exception as e:
            print(f">>>Error attempting to bind {self.back_port}: {e}")
    
    def create_proxy(self):
        '''creates a ZMQ proxy device with the ZMQ front and back port objects so it the device can relay incoming traffic to outgoing '''
        print('>>>creating ZMQ Proxy device...')
        try:
            zmq.proxy(self.frontend, self.backend)
        except zmq.ZMQError as ze:
            print(f">>>Error creating zmq proxy device: {ze}")
        except Exception as e:
            print(f">>>Error creating proxy: {e}")

    def run(self):
        self.create_context()
        self.bind_front()
        self.bind_back()
        self.create_proxy()



def main(option_list):
    
    print(">>>Forwarder running...")
    thr0 = Forwarder(front_port=option_list.pub[0], back_port=option_list.sub[0])
    #thr0.setDaemon(True)
    thr0.start()
    print(threading.current_thread())

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pubPort', '-p', nargs=1, type=int, help='publish port number', default=[10005], dest='pub')
    parser.add_argument('--subPort', '-s', nargs=1, type=int, help='subscribe port number', default=[10006], dest='sub')
    option_list = parser.parse_args()
    return option_list


if __name__ == "__main__":
    option_list = get_arguments()
    main(option_list)

    
