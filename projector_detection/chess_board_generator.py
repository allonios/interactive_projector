import numpy
from screeninfo import get_monitors

def generateChessBoard(square_length = 50):
    width = 1366
    height = 768

    for m in get_monitors():
        width = m.width
        height = m.height

    img = numpy.zeros([height,width])

    edge_height = height % square_length + square_length
    edge_width = width % square_length + square_length

    height_without_edge = height - edge_height
    width_without_edge = width - edge_width

    def checkBlackAndWhite(i, j):
        if i <= square_length or i >= (len(img) - edge_height)  or j <= square_length or j >= (len(img[0]) - edge_width):
            return 1
        return (1 if (int(i/square_length)) % 2 == 0 else -1) * (1 if (int(j/square_length)) % 2 == 0 else -1)

    for i, x in enumerate(img):
        for j, y in enumerate(x):
            img[i,j] = checkBlackAndWhite(i,j)

    return img, height_without_edge / square_length - 1, width_without_edge / square_length - 1, edge_height, edge_width, width, height




# img, width_corners, height_corners, edge_height, edge_width = generateChessBoard(75)
# print(width_corners, height_corners)