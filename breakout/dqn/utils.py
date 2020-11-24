import cv2
import time
import torch
import datetime
import numpy as np 
import matplotlib.pyplot as plt

def plot_results(episodes, scores, filename):
    plt.plot([i for i in range(episodes)], scores)
    plt.xlabel("episodes")
    plt.ylabel("mean scores")
    plt.savefig(filename)

def time_stamp(s):
    t_s = str(datetime.timedelta(seconds=round(s)))
    ts = t_s.split(':')
    return '(' + ts[0] + 'h ' + ts[1] + 'm ' + ts[2] + 's)'

def process_image(image):
    image = np.dot(image[:, :, :], np.array([0.21, 0.72, 0.07])).astype(np.float32)
    image = image[40:-10, :]
    #image = cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
    return torch.Tensor(image)

def create_sequence(history):
    return np.stack(history)