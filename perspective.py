import cv2 as cv
import numpy as np


def sort_rect_points(points):
    """
    Sorts points of rectange in counter-clockwise direction. Starting top left 

    Coods according to:  [x, y]
    +-------> x
    |       1---4
    |       |   |
    |       2---3
    V  y
    """
    if len(points) != 4:
        return None

    def dist_2D(p1, p2):
        return ((p2[0]-p1[0])**2  +  (p2[1]-p1[1])**2)**0.5

    newPoints = np.zeros((4, 2))

    ### Determine top left coordinates
    dList = []  # distance to origin list
    for p in points:
        d = dist_2D([0,0], p)
        dList.append(d)

    dList = np.array(dList)

    points = np.array(points)  # assure it's a numpy array

    order = dList.argsort()  # sort by value
    sortPoints = points[order]  # switch with the same order as dList

    newPoints[0, :] = sortPoints[0]  # add first point

    otherPoints = sortPoints[1:]

    # Coords first point
    x1 = sortPoints[0][0]
    y1 = sortPoints[0][1]

    ### Find third point
    dList = []
    for p in otherPoints:
        d = dist_2D(newPoints[0], p)
        dList.append(d)

    dList = np.array(dList)

    order = dList.argsort()  # sort by value
    sortPoints = otherPoints[order]  # switch with the same order as dList

    newPoints[2, :] = sortPoints[-1]  # add third point

    otherPoints = sortPoints[:-1]

    ### Search for second point

    # search point to the left
    secondPFound = False
    for i, p in enumerate(otherPoints):
        if p[0] <= x1:  # point to the left was found
            newPoints[1, :] = p
            secondPFound = True
            otherPoints = np.delete(otherPoints, i, 0)
            break
    if secondPFound == True:  # add third point
        newPoints[3, :] = otherPoints[0]
        return newPoints  # RETRUN since all points were found

    ## no second point was found to the left of the first point
    # select the point with the lower y-value as second point
    if otherPoints[0][1] > otherPoints[1][1]:
        newPoints[1, :] = otherPoints[0]
        newPoints[3, :] = otherPoints[1]
    else:
        newPoints[1, :] = otherPoints[1]
        newPoints[3, :] = otherPoints[0]
    
    newPoints = np.array(newPoints, dtype='float32')
    return newPoints


def warp_image(img, cornerpointsSquare, scale=1):
    """Warps image to accuratly show a square.
    :img:  image as numpy array
    :cornerpointsSquare:  points of the square in the image (no particular order needed)
    :scale:  scales the image (scale > 1: Zoom In; scale < 1: Zoom out)
    :returns:  warped image with the same shape
    """

    y, x, _ = img.shape

    # sort to make a valid imagewarp
    cornerpointsSquare = sort_rect_points(cornerpointsSquare)

    # calc cornerpoints in destination image (assuming x/y > 1, e.g. 16/9 or 4/3)
    halfSideLength = np.floor((min(x, y) / 2) * scale)

    xMid = max(x, y) // 2
    yMid = min(x, y) // 2

    destPoints = np.array([
        [xMid - halfSideLength, yMid - halfSideLength],
        [xMid - halfSideLength, yMid + halfSideLength],
        [xMid + halfSideLength, yMid + halfSideLength],
        [xMid + halfSideLength, yMid - halfSideLength]
    ], dtype='float32')

    cornerpointsSquare = np.array(cornerpointsSquare, dtype='float32')

    # create the image warp matrix
    warpMtx = cv.getPerspectiveTransform(cornerpointsSquare, destPoints)

    newImgSize = (x, y)

    warpImg = cv.warpPerspective(img, warpMtx, newImgSize)
    
    return warpImg



########## For Testing only ############

def _test_sorting():
    points = np.array([
                [40, 264],
                [371,289],
                [300,93],
                [91,122],
                ], dtype='float32')

    points = np.array([
                [351,215],
                [162,311],                
                [85,160],
                [274,64]], dtype='float32')

    sorted = sort_rect_points(points)

def _main():
    path = "ideal3.jpg"

    img = cv.imread(path)

    # Points ideal2 & ideal3:
    """
    91,122         300,93
    40,264         371,289

    Order:
    1 4
    2 3

    """
    # Points ideal:
    """
    85,160          274,64
    162,311         351,215
    """


    points1 = np.array([
                [85,160],
                [162,311],
                [351,215],
                [274,64]], dtype='float32')

    points = np.array([
                [91,122],
                [40, 264],
                [371,289],
                [300,93]
                ], dtype='float32')


    l = 400
    r = 800

    destPoints = np.array([
                [l, l],
                [l, r],
                [r, r],
                [r, l]], dtype='float32')

    warpMtx = cv.getPerspectiveTransform(points, destPoints)

    newImgSize = (1000, 1000)

    warpImg = cv.warpPerspective(img, warpMtx, newImgSize)

    cv.imshow("original", img)
    cv.imshow("warped", warpImg)
    cv.waitKey(0)


if __name__ == "__main__":
    _main()
    _test_sorting()