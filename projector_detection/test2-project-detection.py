import urllib.request
import cv2
import numpy as np
import time
import argparse
import time


def rescaleFrame(frame, scale=0.25):
	width = int(frame.shape[1] * scale)
	height = int(frame.shape[0] * scale)
	dimensions = (width, height)

	return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

URL = "http://192.168.1.100:8080/shot.jpg"

	# !!! IMPORTANT, set the nx, ny according the calibration chessboard pictures.
nx = 8
ny = 8

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d pionts in image plane.

# Step through the list and search for chessboard corners
while True:
	img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
	img = cv2.imdecode(img_arr, -1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	print(img.shape[0])

	cv2.imshow("img", gray)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
		cv2.waitKey(500)
	# cv2.imshow(img)

cv2.destroyAllWindows()

# Get image size
# img_size = (img.shape[1],img.shape[0])

# Do camera calibration given object points and image points
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)