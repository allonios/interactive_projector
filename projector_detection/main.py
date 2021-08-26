import numpy
from numpy.linalg import inv
import urllib.request
import cv2
from screeninfo import get_monitors

def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def generateChessBoard(square_length = 50):
    width = 1366
    height = 768

    for m in get_monitors():
        width = m.width
        height = m.height

    img = numpy.zeros([height,width], dtype="float32")

    edge_height = height % square_length
    edge_width = width % square_length

    height_without_edge = height - edge_height
    width_without_edge = width - edge_width

    def checkBlackAndWhite(i, j):
        if i <= square_length or i >= (len(img) - edge_height)  or j <= square_length or j >= (len(img[0]) - edge_width):
            return 1
        return (-1 if (int(i/square_length)) % 2 == 0 else 1) * (1 if (int(j/square_length)) % 2 == 0 else -1)

    for i, x in enumerate(img):
        for j, y in enumerate(x):
            img[i,j] = 0 if (checkBlackAndWhite(i,j) == -1) else 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img, (height_without_edge / square_length) - 2, (width_without_edge / square_length) - 2, edge_height, edge_width, width, height, square_length


def groundProjectPoint(image_point, camera_matrix, rotMat, tvec, z = 0.0):
    camMat = numpy.asarray(camera_matrix)
    iRot = inv(rotMat)
    iCam = inv(camMat)

    uvPoint = numpy.ones((3, 1))

    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = numpy.matmul(numpy.matmul(iRot, iCam), uvPoint)
    tempMat2 = numpy.matmul(iRot, tvec)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = numpy.matmul(iRot, (numpy.matmul(s * iCam, uvPoint) - tvec))

    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z

    return wcPoint



URL = "http://192.168.1.101:8080/shot.jpg"

# Generating Chessboard ...
chess_board_img, width_corners, height_corners, edge_height, edge_width, width, height, length = generateChessBoard(100)
# Detecting Chess board ...
objp = numpy.zeros((int(width_corners) * int(height_corners), 3), numpy.float32)
objp[:,:2] = numpy.mgrid[0:int(height_corners),0:int(width_corners)].T.reshape(-1,2)


print("objp.shape = ",objp.shape)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

print(width_corners, height_corners)
while True:
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", chess_board_img)

    img_arr = numpy.array(
        bytearray(urllib.request.urlopen(URL).read()), dtype=numpy.uint8)
    img = cv2.imdecode(img_arr, -1)
    # img = rescaleFrame(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (int(height_corners),int(width_corners)), None)
    if ret == True:
        print("Find it")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (int(height_corners),int(width_corners)), corners2, ret)

        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        dist_coeffs = numpy.zeros((4, 1))

        objpoints = numpy.array(objpoints[0])
        imgpoints = numpy.array(imgpoints[0], dtype="double")
        imgpoints = numpy.reshape(imgpoints,(imgpoints.shape[0],2))

        success, rotation_vector, translation_vector = cv2.solvePnP(objpoints, imgpoints, cameraMatrix, dist_coeffs,
                                                                    flags=0)
        rotMat, _ = cv2.Rodrigues(rotation_vector)

        pixel = (250, 250)
        real_world_coordinates = groundProjectPoint(pixel, cameraMatrix, rotMat, translation_vector)
        real_world_point = (int((real_world_coordinates[0] + 2) * length), int((real_world_coordinates[1] + 2) * length))
        print("real_world_point = ",real_world_point)
        print("real_world_coord = ",real_world_coordinates)
        cv2.circle(img, pixel, 3, (0, 255, 255), -1)
        cv2.circle(chess_board_img, real_world_point, 3, (0, 0, 255), -1)



        # just for testing the coordinates
        # new_point, jacobian = cv2.projectPoints((-2., -2., 0.), rotation_vector,
        #                                                translation_vector, cameraMatrix, dist_coeffs)
        # point2 = (int(new_point[0][0][0]), int(new_point[0][0][1]))
        # cv2.circle(img, point2, 3, (255, 0, float(255), -1))

        for p in imgpoints:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)




        # h, w = img.shape[:2]
        # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

        # dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]

        # cv2.imwrite('caliResult1.png', dst)
        # cv2.imwrite('dist.png', dist)
        cv2.imwrite('chessboard.png', chess_board_img * 255)
        cv2.imwrite('base.png', img)
        break

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


