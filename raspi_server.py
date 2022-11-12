"""
This is a program for receiving video and predictions from the Raspberry Pi.
The program(backend) is deployed on the Raspberry Pi, and the server(frontend) is deployed on the PC.
"""

import argparse
import sys
import socket
import yaml

import cv2
import numpy as np

from utils import Cmapper

argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, default="configs/raspi_server.yml", help="path to config file")
args = argparser.parse_args()

# read yml file
print("config file: ", args.config)
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

address = config["address"]
port = config["port"]

# cmaper
# cmaper is used to convert the prediction to a color image
cmapper = Cmapper(cmap="plasma", maximum=255., minimum=0.)

# socket
try:
    # create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((address, port))
    s.listen(1)
except socket.error as e:
    print(e)
    exit(-1)
except:
    print("Unexpected error:", sys.exc_info()[0])
    exit(-1)

# start receiving and displaying
conn, addr = s.accept()
print("connected by ", addr)

# main loop
while True:
    # receive video and prediction
    try:
        data = conn.recv(1024)
        if not data:
            break

        # receive an array containing frame and pred
        # the first 4 bytes is the length of frame
        # the second 4 bytes is the length of pred
        # the rest is the frame and pred
        frame_len = int.from_bytes(data[:4], byteorder="big")
        pred_len = int.from_bytes(data[4:8], byteorder="big")
        total_len = frame_len + pred_len

        # receive the rest of data
        while len(data) < total_len + 8:
            data += conn.recv(1024)
        # slice the data
        frame = data[8:8+frame_len]
        pred = data[8+frame_len:8+frame_len+pred_len]
        # decode frame and pred
        frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), -1)
        pred = cv2.imdecode(np.frombuffer(pred, dtype=np.uint8), -1)
        # colorize pred
        pred = cmapper(pred)
        # interpolate pred to the size of frame
        pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    except socket.error as e:
        print(e)
        break
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break
    except:
        print("Unexpected error:", sys.exc_info())
        break

    # show video and prediction
    cv2.imshow("frame", frame)
    cv2.imshow("pred", pred)

    if cv2.waitKey(1) == ord("q"):
        break
