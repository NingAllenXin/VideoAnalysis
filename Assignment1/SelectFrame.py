import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('output.mp4')

success,image = cap.read()

count = 0;
while success:
	success,image = cap.read()
	cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
	count += 1
