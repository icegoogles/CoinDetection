import cv2
import numpy as np
import random as rand
import math



def points_to_hess_equ(points):
    """
    :returns: list in form of [[r_0, theta_0], ...,[r_n, theta_n]]

    r can be negative!
    """

    storageList = []

    for line in points:
        for x1, y1, x2, y2 in line:
            if x1 == x2: 
                theta = 0
            else:
                theta = math.atan((y2 - y1) / (x2 - x1)) + math.pi/2
            
            r = x1*math.cos(theta) + y1*math.sin(theta)

            storageList.append([r, theta])

def calc_point_intersec(points, imgDim=None, angleRange=[0, 80]):
    vectorList = []

    # Make line in form of vector equation
    for line in points:
        for x1, y1, x2, y2 in line:
            r = [x1, y1]
            a = [x2-x1, y2-y1]

            vectorList.append([r, a])

    length = len(vectorList)
    pointList = []

    dimTest = False
    if imgDim is not None: dimTest = True

    def is_in_image(x, y, imgDim=imgDim):  # function to test, if the point is inside the image
        if x >= 0 and x < imgDim[0] and y >= 0 and y < imgDim[1]:
            return True
        return False

    for i, equ in enumerate(vectorList):
        for j in range(i+1, length):
            r1, a1 = equ[0], equ[1]
            r2, a2 = vectorList[j][0], vectorList[j][1]

            r1x, r2x = r1[0], r2[0]
            r1y, r2y = r1[1], r2[1]
            a1x, a2x = a1[0], a2[0]
            a1y, a2y = a1[1], a2[1]

            # Check if lines are in the given anglerange
            top = np.dot(a1, a2)
            bot = np.linalg.norm(a1) * np.linalg.norm(a2)

            fraction = np.clip(top/bot, -1, 1)  # check acos range
                
            angle = math.degrees(abs(math.acos(fraction)))  # angle between lines (in DEG)

            if not (angle >= angleRange[0] and angle <= angleRange[1]): continue

            # Calculate intersection point
            if 0 == (a1x*a2y - a2x*a1y): continue  # check for division by 0

            lamb1 = (a2x*r1y - a2y*r1x - a2x*r2y + a2y*r2x)/(a1x*a2y - a2x*a1y)


            x = round(r1x + lamb1*a1x)
            y = round(r1y + lamb1*a1y)

            if dimTest == True and not is_in_image(x, y, imgDim): continue

            pointList.append([x, y])
            
    return pointList



def points_to_lin_equ(points):
    """
    :returns: list in form of [[m_0, b_0], ...,[m_n, b_n]]
    """

    storageList = []

    for line in points:
        for x1, y1, x2, y2 in line:
            if x1 == x2: x2 += 0.00001  # approximate lines parallel to the y-axis

            m = (y2 - y1) / (x2 - x1)  # calculate slope
            b = y1 - m*x1  # calculate y-axis offset

            storageList.append([m, b])

    return storageList

def calc_intersections(linEquations, imgDim=None, angleRange=[0, 91]):
    """
    :linEquations: in the form of [[m_0, b_0], ..., [m_0, b_0]]
    :imgDim: list ([x,y]) with dimensions of the image to only save points inside th image
    :angleRange: range of angles (in DEGREES) where intersections should be counted (e.g. [30, 70])
    :returns: list of points in the form [[x_0, y_0], ..., [x_n, y_n]]
    """
    length = len(linEquations)
    pointList = []

    dimTest = False
    if imgDim is not None: dimTest = True

    def is_in_image(x, y, imgDim=imgDim):
        if x >= 0 and x < imgDim[0] and y >= 0 and y < imgDim[1]:
            return True
        return False

    for i, equ in enumerate(linEquations):
        for j in range(i+1, length):
            m1, b1 = equ[0], equ[1]
            m2, b2 = linEquations[j][0], linEquations[j][1]

            # Check for anglerange
            angle = abs(math.degrees(abs(math.atan(m1)) - abs(math.atan(m2))))
            if not (angle >= angleRange[0] and angle <= angleRange[1]): continue

            m = m1 - m2
            b = b2 - b1

            if m == 0: continue  # do not go further if lines are parallel

            # calculate intersection point
            x = round(b / m)
            y = round(m1*x + b1)

            if dimTest == True and not is_in_image(x, y, imgDim): continue

            pointList.append([x, y])
    
    return pointList

def find_corners(image):

    lines = findlines(image)
    if lines is None: return None

    y, x, _ = image.shape

    points = calc_point_intersec(lines, imgDim=[x,y], angleRange=[45, 135])

    mainPoints = calc_prominent_points(points, 4)

    if len(mainPoints) < 4: return None  # assure 4 cornerpoints

    # convert to list
    
    tempList = []

    for point in mainPoints:
        tempList.append([int(point[0]), int(point[1])])

    mainPoints = tempList

    return mainPoints

def draw_corners(image, points, color=(0,255,0), pointsize=10):
    for p in points:
        cv2.circle(image, (p[0], p[1]), pointsize, color, -1)
    return image


def draw_all_intersections(image):
    
    lines = findlines(image)
    if lines is None: return image  # return original image if no lines were found

    y, x, _ = image.shape

    # Find intersection points
    # linEquations = points_to_lin_equ(lines)
    # points = calc_intersections(linEquations, imgDim=[x,y], angleRange=[30, 91])
    points = calc_point_intersec(lines, imgDim=[x,y], angleRange=[45, 135])

    mainPoints = calc_prominent_points(points, 4)

    # This can be prevented when using the `imgDim` parameter while calculating intersections
    maxValue = 2000000000  # maximum value in array prevents an error while converting to a C-long
    points = np.clip(points, -maxValue, maxValue)  # clip array to maximum value

    # draw points
    for p in points:
        cv2.circle(image, (p[0], p[1]), 3, (0,0,255), -1)

    for p in mainPoints:
        cv2.circle(image, (p[0], p[1]), 7, (0,255,0), -1)

    return image

def calc_prominent_points(points, k):
    """
    :points: given points in the form [[x_0, y_0], ..., [x_n, y_n]]
    :k: numer of prominent points
    :returns: prominent points
    """
    if len(points) <= 1: return points


    ### K-Means
    if len(points) < k: return points

    points = np.array(points)  # convert to numpy-array

    points = points.astype(np.float32)  # convert to float for use in kmeans

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, bestLabels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # If the cleanup isn't nessecary the centers can be just returned here
    # return centers

    ### Cleanup results by removing outliers
    def dist_2D(p1, p2):
        return ((p2[0]-p1[0])**2  +  (p2[1]-p1[1])**2)**0.5

    newCenters = []

    for i, centpt in enumerate(centers):
        distanceList = []
        pointList = []
        for j, label in enumerate(bestLabels):

            if label[0] != i: continue  # skip if point does not belong to the label/group

            dist = dist_2D(centpt, points[j])
            # write to lists
            distanceList.append(dist)
            pointList.append(j)


        # convert to numpy-array
        dists =  np.array(distanceList)
        pointNbr = np.array(pointList)

        # sort distances (for median)
        order = dists.argsort()
        sortDists =  dists[order]
        sortPoints = pointNbr[order]  # sort points the same way as distance

        midPos = len(sortPoints) // 2

        bestPoint = sortPoints[midPos]

        newCenters.append(points[bestPoint])
    
    return newCenters



def main():
    img = cv2.imread('ideal2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("asass", thresh1)

    gray = thresh1


    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    ###########

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    ##############


    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    print(len(lines))


    # # Draw Lines on image
    # def randCol():
    #     return (rand.randint(0,255), rand.randint(0,255), rand.randint(0,255))

    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(line_image, (x1,y1), (x2,y2), randCol(), 5)

    # Find intersection points
    linEquations = points_to_lin_equ(lines)
    points = calc_intersections(linEquations)

    # draw points
    point_image = np.copy(img)
    for p in points:
        cv2.circle(point_image, (p[0], p[1]), 3, (0,0,255), -1)

    ################


    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)


    cv2.imshow("aa", lines_edges)
    cv2.imshow("asa", line_image)
    cv2.imshow("kl", point_image)
    cv2.waitKey(0)


def findlines(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # make binary image
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(thresh1,(kernel_size, kernel_size),0)

    ###########

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    ##############


    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 90  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    return lines

   

if __name__ == "__main__":
    #main()

    p = [[[0, 1, 1, 0]], [[0, -1, 2, 0]]]
    #points_to_hess_equ(p)
    ans = calc_point_intersec(p)
    pass
