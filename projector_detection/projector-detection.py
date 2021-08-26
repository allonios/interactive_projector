import urllib.request
import cv2 as cv
import numpy as np
import time
import argparse
import time

from screeninfo import get_monitors
for m in get_monitors():
    print(str(m))


def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


URL = "http://192.168.1.102:8080/shot.jpg"
chess_board_img = "chessboard03.png"
prev_time = time.time()

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    chessboard = cv.imread(chess_board_img)
    print(chessboard.shape[0], chessboard.shape[1], chessboard.shape[2])
    #853 1137 shape of chess board image
    cv.namedWindow("window", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow("window", chessboard)

    prev_time = time.time()
    img_arr = np.array(
        bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    img = cv.imdecode(img_arr, -1)
    img = rescaleFrame(img)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    # corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if ret == True:
        print()
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print(corners2)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv.waitKey(1000)
        # break

    cv.imshow('img', img)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

