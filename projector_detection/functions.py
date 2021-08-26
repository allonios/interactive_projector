from screeninfo import get_monitors
import numpy
from numpy.linalg import inv

import urllib.request
import cv2

URL = "http://192.168.46.114:8080/shot.jpg"


def generate_chess_board(square_length=50):
    width = 1366
    height = 768

    for m in get_monitors():
        width = m.width
        height = m.height

    img = numpy.zeros([height, width], dtype="float32")

    edge_height = height % square_length
    edge_width = width % square_length

    height_without_edge = height - edge_height
    width_without_edge = width - edge_width

    def checkBlackAndWhite(i, j):
        if i <= square_length or i >= (len(img) - edge_height) or j <= square_length or j >= (len(img[0]) - edge_width):
            return 1
        return (-1 if (int(i / square_length)) % 2 == 0 else 1) * (1 if (int(j / square_length)) % 2 == 0 else -1)

    for i, x in enumerate(img):
        for j, y in enumerate(x):
            img[i, j] = 0 if (checkBlackAndWhite(i, j) == -1) else 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img, (height_without_edge / square_length) - 2, (
            width_without_edge / square_length) - 2, edge_height, edge_width, width, height, square_length


def show_chess_board(chess_board_img):
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", chess_board_img)


def calibrate(source_function):
    chess_board_img, width_corners, height_corners, edge_height, edge_width, width, height, length = generate_chess_board(
        100)

    objp = numpy.zeros((int(width_corners) * int(height_corners), 3), numpy.float32)
    objp[:, :2] = numpy.mgrid[0:int(height_corners), 0:int(width_corners)].T.reshape(-1, 2)
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane.

    while True:
        show_chess_board(chess_board_img)
        img = source_function()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (int(height_corners), int(width_corners)), None)
        if ret == True:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners)
            _, camera_matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None,
                                                            None)
            dist_coeffs = numpy.zeros((4, 1))

            # reshaping
            object_points = numpy.array(object_points[0])
            image_points = numpy.array(image_points[0], dtype="double")
            image_points = numpy.reshape(image_points, (image_points.shape[0], 2))

            success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix,
                                                                        dist_coeffs,
                                                                        flags=0)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            return True, camera_matrix, rotation_matrix, translation_vector, rotation_vector, length, width, height, width_corners, height_corners, edge_width, edge_height, img
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break


def get_real_coordinates(pixel, camera_matrix, rotation_matrix, translation_vector, square_length):
    z = 0
    camMat = numpy.asarray(camera_matrix)
    iRot = inv(rotation_matrix)
    iCam = inv(camMat)

    uvPoint = numpy.ones((3, 1))

    # Image point
    uvPoint[0, 0] = pixel[0]
    uvPoint[1, 0] = pixel[1]

    tempMat = numpy.matmul(numpy.matmul(iRot, iCam), uvPoint)
    tempMat2 = numpy.matmul(iRot, translation_vector)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = numpy.matmul(iRot, (numpy.matmul(s * iCam, uvPoint) - translation_vector))

    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z

    return (int((wcPoint[0] + 2) * square_length), int((wcPoint[1] + 2) * square_length))


def check_if_inside_the_screen(screen_width, screen_height, pixel, camera_matrix, rotation_matrix, translation_vector,
                               square_length):
    real_coordinates = get_real_coordinates(pixel, camera_matrix, rotation_matrix, translation_vector, square_length)
    if (real_coordinates[0] < 0 or real_coordinates[1] < 0 or real_coordinates[0] > screen_height or real_coordinates[
        0] > screen_width):
        return False, real_coordinates
    return True, real_coordinates


def calibrate_from_chess_baord(source_function):
    calibrated, camera_matrix, rotation_matrix, translation_vector, _, square_length, width, height, _, _, _, _, _ = calibrate(
        source_function)
    return lambda point: check_if_inside_the_screen(width, height, point, camera_matrix, rotation_matrix,
                                                    translation_vector, square_length)


def calibrate_from_configuration(camera_matrix, rotation_matrix, translation_vector, square_length, width, height):
    return lambda point: check_if_inside_the_screen(width, height, point, camera_matrix, rotation_matrix,
                                                    translation_vector, square_length)


def getZoomedImage(img, corners_width, corners_hegiht, edge_width, edge_height, square_length, rotation_vector,
                   translation_vector, camera_matrix, margin_right=0, margin_left=0, margin_top=0, margin_bottom=0):
    dist_coeffs = numpy.zeros((4, 1))

    edge_width_coordinate = edge_width / square_length
    edge_height_coordinate = edge_height / square_length

    top_left_point, _ = cv2.projectPoints((-2., -2., 0.), rotation_vector, translation_vector, camera_matrix,
                                          dist_coeffs)
    top_left_point = (int(top_left_point[0][0][0]), int(top_left_point[0][0][1]))
    cv2.circle(img, top_left_point, 3, (255, 0, float(255), -1))

    top_right_point, _ = cv2.projectPoints((-2., corners_width + edge_width_coordinate, 0.), rotation_vector,
                                           translation_vector,
                                           camera_matrix, dist_coeffs)
    top_right_point = (int(top_right_point[0][0][0]), int(top_right_point[0][0][1]))
    cv2.circle(img, top_right_point, 3, (255, 0, float(255), -1))

    bottom_left_point, _ = cv2.projectPoints((corners_hegiht + edge_height_coordinate, -2., 0.), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
    bottom_left_point = (int(bottom_left_point[0][0][0]), int(bottom_left_point[0][0][1]))
    cv2.circle(img, bottom_left_point, 3, (255, 0, float(255), -1))

    bottom_right_point, _ = cv2.projectPoints(
        (corners_hegiht + edge_height_coordinate, corners_width + edge_width_coordinate, 0.),
        rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    bottom_right_point = (int(bottom_right_point[0][0][0]), int(bottom_right_point[0][0][1]))
    cv2.circle(img, bottom_right_point, 3, (255, 0, float(255), -1))

    max_y = max(bottom_right_point[0], bottom_left_point[0], top_left_point[0], top_right_point[0]) + margin_bottom
    min_y = min(bottom_right_point[0], bottom_left_point[0], top_left_point[0], top_right_point[0]) - margin_top

    max_x = max(bottom_right_point[1], bottom_left_point[1], top_left_point[1], top_right_point[1]) + margin_right
    min_x = min(bottom_right_point[1], bottom_left_point[1], top_left_point[1], top_right_point[1]) - margin_left

    crop_img = img[min_x:max_x, min_y: max_y]
    print(crop_img.shape, min_y, max_y, min_x, max_x)
    cv2.imshow("resizedImage", img)
    if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
        cv2.imshow("cropedImage", crop_img)

    cv2.waitKey()


def mobile_source_function():
    img_arr = numpy.array(
        bytearray(urllib.request.urlopen(URL).read()), dtype=numpy.uint8)
    return cv2.imdecode(img_arr, -1)


# func = calibrate_from_chess_baord(mobile_source_function)
# inside, real_coordinates = func((100, 100))
#
# print(inside, real_coordinates)

calibrated, camera_matrix, _, translation_vector, rotation_vector, square_length, _, _, width_corners, height_corners, edge_width, edge_height, img = calibrate(
    mobile_source_function)

getZoomedImage(img, width_corners, height_corners, edge_width, edge_height, square_length, rotation_vector,
               translation_vector, camera_matrix, 20, 10, 30, 40)
# cv2.circle(img, point, 3, (0, 255, 255), -1)
# cv2.circle(chess_img, real_point, 3, (0, 255, 255), -1)
#
# cv2.imwrite("chessboard.png", chess_img)
# cv2.imwrite("result.png", img)
