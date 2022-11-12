"""
This is a program for sending video and predictions to the server.
The program(backend) is deployed on the Raspberry Pi, and the server(frontend) is deployed on the PC.
"""

import argparse
import threading
import time
import socket
import yaml

import cv2
import albumentations as A
import albumentations.pytorch as AP

import torch


argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, default="configs/raspi.yml", help="path to config file")
args = argparser.parse_args()

# read yml file
print("config file: ", args.config)
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

device = config["device"]
torch_num_threads = config["torch_num_threads"]
resolution = config["resolution"]

fps = config["fps"]
model_path = config["model_path"]

address = config["address"]
port = config["port"]

# camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
cap.set(cv2.CAP_PROP_FPS, fps)

# pytorch configs
torch.set_num_threads(torch_num_threads)
torch.set_grad_enabled(False)

# transforms
transforms = A.Compose([
    A.Resize(*resolution, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    AP.ToTensorV2()
])

# model
model = torch.load(model_path)
model = model.to(device)
model.eval()


# predicting function
def predict(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(image=img)["image"]
    img = img.unsqueeze(0).to(device)
    preds = model(img)
    preds = preds[0, 0] / 10. * 255.
    preds = preds.to("cpu", dtype=torch.uint8).numpy()
    return preds


# build socket
print("Address: ", address, "  Port: ", port)
while True:
    try:
        # create socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect to server
        s.connect((address, port))
        print("Connected to server")
        print("=========================================")
        break
    except ConnectionRefusedError:
        print("Waiting for connection...")
        time.sleep(2)
    except:
        print("Unknown error")
        exit(-1)

# build 2 threads and a queue
# one thread for capturing video, predicting and storing in queue
# one thread for sending video and predictions

# queue for storing predictions
queue = []

# semaphore for synchronizing threads
mutex = threading.Semaphore(1)
empty = threading.Semaphore(0)

# flog for stopping threads
flag = True


# thread for capturing video
def capture():
    global flag
    while flag:
        # capture frame from camera
        ret, frame = cap.read()
        if ret:
            # predict depth
            pred = predict(frame)
            # P(mutex)
            mutex.acquire()
            # add frame and pred to buffer queue
            queue.append([frame, pred])
            # V(empty)
            empty.release()
            # V(mutex)
            mutex.release()
        else:
            # if failed to capture frame, stop the thread
            flag = False
    else:
        print("Capture thread stopped")

# thread for sending video and predictions
def send():
    global flag
    while flag:
        # P(empty)
        empty.acquire()
        # P(mutex)
        mutex.acquire()
        # get frame and pred from buffer queue
        frame, pred = queue.pop(0)
        # V(mutex)
        mutex.release()
        # encode frame and pred to bytes for sending
        frame = cv2.imencode(".jpg", frame)[1].tobytes()
        pred = cv2.imencode(".jpg", pred)[1].tobytes()

        # build an array to store frame and pred
        # the first 4 bytes is the length of frame
        # the second 4 bytes is the length of pred
        # the rest is the frame and pred
        data = len(frame).to_bytes(4, byteorder="big") + \
               len(pred).to_bytes(4, byteorder="big") + \
               frame + pred

        try:
            # try to send all data
            s.sendall(data)
        except:
            # if failed to send all data, stop the thread
            print("Error sending data")
            flag = False
    else:
        print("Send thread stopped")

# start threads
# press "ctrl + c" to stop
t_predict = threading.Thread(target=capture)
t_send = threading.Thread(target=send)
t_predict.start()
t_send.start()

# main thread waits for threads to stop
try:
    while flag:
        time.sleep(1)
    else:
        raise KeyboardInterrupt
except KeyboardInterrupt:
    # stop threads
    flag = False
    # release camera
    cap.release()
    # close socket
    s.close()
    # kill threads
    t_predict.join()
    t_send.join()
