# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=256,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
whiteLower = np.array([45,40,180], dtype=np.uint8)
whiteUpper = np.array([80,70,230], dtype=np.uint8)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam

camera = cv2.VideoCapture('cs6327-a2.mp4')
count = 0
time = 1

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=600)
	#blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, whiteLower, whiteUpper)
	#mask = cv2.erode(mask, None, iterations=2)
	#mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 5:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius) + 3,
				(0, 255, 255),2)
			#cv2.circle(frame, center, 5, (0, 0, 255), -1)

	if len(pts) >= 1:
		x = (center[0] - pts[0][0]) ** 2
		y = (center[1] - pts[0][1]) ** 2
		print "Speed : {0} pixels per second".format(((x + y) ** 0.5)*30)

	#if len(pts) >= 31 :
	#	x = (center[0] - pts[count][0]) ** 2
	#	y = (center[1] - pts[count][1]) ** 2

		#print "No. {0}".format(count)
	#	count += 1
		#print "Speed : {0} pixel/second".format((x + y) ** 0.5)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.5)
		cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), thickness)
		cv2.line(mask, pts[i - 1], pts[i], (255, 255, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	cv2.imshow('mask', mask)

	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break



# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
