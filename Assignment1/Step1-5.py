import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while(cap.isOpened()):
    # Take each frame
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.flip(frame, 0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

    # Number of frames to capture
    num_frames = 6;

    print "Capturing {0} frames".format(num_frames)

    # Start time
    start = time.time()

    # Grab a few frames
    for i in xrange(0, num_frames):
        ret, frame = cap.read()

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print "Time taken : {0} seconds".format(seconds)

    # Calculate frames per second
    fps = num_frames / seconds;
    print "Estimated frames per second : {0}".format(fps);



cap.release()
out.release()
cv2.destroyAllWindows()