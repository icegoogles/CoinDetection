from math import ceil, floor, atan2, degrees
import cv2
import cv2 as cv
import numpy as np
from PIL import ImageTk, Image

import config as cfg  # load config


def start_bv(frame, scaleInPXperMM):
    if scaleInPXperMM is None:
        print("Please calibrate the setup first!")
        return frame, np.zeros(10)


    # minimum center distance
    minCDist = cfg.minDiameter / 2 * scaleInPXperMM
    if cfg.coinsCanOverlap == False:
        minCDist = cfg.minDiameter * scaleInPXperMM
    

    minRadius = int(np.floor(cfg.minDiameter * scaleInPXperMM / 2))
    maxRadius = int(np.ceil(cfg.maxDiameter * scaleInPXperMM / 2))

    print("minD: ", minRadius/scaleInPXperMM * 2, "\nmaxD: ", maxRadius/scaleInPXperMM * 2)

    ### Circledetection  /////////////////////////////////////////////////////////////////////
    
    b, g, r = cv2.split(frame)

    m1 = 1.0/np.sqrt(6) * (2*r - g - b)
    m2 = 1.0/np.sqrt(2) * (g - b)


    cv2.imwrite("frame.jpg", frame)

    #### Green extraction
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hue, sat, val = cv2.split(hsv)

    hue_blur = cv2.medianBlur(hue, 5)
    
    cv2.imshow("HUE", hue_blur)
    cv2.imwrite("HUE.jpg", hue_blur)

    # lower = round(lower / 360 * 255)
    # upper = round(upper / 360 * 255)

    #### Set limits
    # convert [0, 359] to [0, 180]
    hueAngleMean = round(cfg.hueAngleMean / 2)
    hueTolerance = round(cfg.hueTolerance / 2)

    upperHue = hueAngleMean + hueTolerance
    lowerHue = hueAngleMean - hueTolerance

    if upperHue >= 180:  # leaving upper bounds
        mask1 = ~cv2.inRange(hue_blur, lowerHue, 179)
        mask2 = ~cv2.inRange(hue_blur, 0, upperHue - 180)
        mask = mask1 | mask2
    elif lowerHue < 0:  # leaving lower bounds
        mask1 = ~cv2.inRange(hue_blur, lowerHue + 180, 179)
        mask2 = ~cv2.inRange(hue_blur, 0, upperHue)
        mask = mask1 | mask2
    else:  # normal case
        mask = ~cv2.inRange(hue_blur, lowerHue, upperHue)
        

    # mask = cv2.inRange(hue_blur, lower, upper)
    # mask = ~mask

    #cv2.imshow("mask", mask)

    iterations = 2
    kernSize = 7  # odd number
    kernSizeHalf = int(np.floor(kernSize/2))

    # round kernel
    kernel = np.zeros((kernSize, kernSize), np.uint8)
    kernel = cv2.circle(kernel, (kernSizeHalf, kernSizeHalf), kernSizeHalf, (1), -1)

    cv2.imwrite("mask1.jpg", mask)

    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    #cv2.imshow("dilated", dilated)
    cv2.imwrite("dilated.jpg", dilated)

    eroded = cv2.erode(dilated, kernel, iterations=iterations)

    #cv2.imshow("eroded", eroded)
    cv2.imwrite("eroded.jpg", eroded)

    eroded_blur = cv2.medianBlur(eroded, 5)

    #cv2.imshow("eroded_blur", eroded_blur)
    cv2.imwrite("eroded_blur.jpg", eroded_blur)

    edges = cv2.Canny(eroded_blur, 10, 240, 3)

    cv2.imshow("edges", edges)
    cv2.imwrite("edges.jpg", edges)

    #dilated = cv2.dilate(edges, kernel, iterations=1)
    #cv2.imshow("dilated", dilated)

    # HoughCircles
    # orgignal 250, 30
    circles	= cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minCDist, param1=50, param2=15, minRadius=minRadius, maxRadius=maxRadius)
    if circles is None:
        print("No circles found!")
        return frame, np.zeros(10)


    coinList = check_for_coins(circles, frame, scaleInPXperMM)

    print(coinList)

    
    #imgWCircles = draw_circles(circles, frame)
    imgWCircles = draw_circles(circles, frame, coinType=coinList, scaleInPXperMM=scaleInPXperMM)

    # moneyarray[2€, 1€, 50ct, 20ct, 10ct, 5ct, 2ct, 1ct, Falschgeld, Gesamtbetrag in €]
    money = [0, 0, 0, 0, 0, 0, 0, 0, 0, None]

    for i in coinList:
        if i == '2e':
            money[0] += 1
        elif i == '1e':
            money[1] += 1
        elif i == '50c':
            money[2] += 1
        elif i == '20c':
            money[3] += 1
        elif i == '10c':
            money[4] += 1
        elif i == '5c':
            money[5] += 1
        elif i == '2c':
            money[6] += 1
        elif i == '1c':
            money[7] += 1
        else:
            money[8] += 1


    money[9] = calc_amount(money)
    
    return imgWCircles, money


def draw_circles(circles, img, coinType=None, scaleInPXperMM = 0):
    newimg = img.copy()

    circles	= np.uint16(np.around(circles))

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # fontScale
    fontScale = 0.7

    # Line thickness
    thickness = 2

    for n, i in enumerate(circles[0, :]):
        # draw the outer circle
        cv2.circle(newimg, (i[0], i[1]), i[2], (239, 0, 255), 2)
        #	drawthe	center	of	the	circle
        cv2.circle(newimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        if coinType is not None:
            if coinType[n] is None:
                coinType[n] = 'None'
            
            diamStr = str(round(2 * i[2] / scaleInPXperMM, 2)) + ' mm'

            cv2.putText(newimg, coinType[n] + ' ' + str(n), (i[0], i[1]), font, fontScale, 
                 (255, 0, 0), thickness, cv2.LINE_AA, False)

            cv2.putText(newimg, diamStr, (i[0] - 70, i[1] + int(30 * fontScale)), font, fontScale, 
                 (0, 0, 255), thickness, cv2.LINE_AA, False)

    return newimg

def calc_amount(money_vec):
    """`monex_vec` list or np.array with number of coins: starting with 2€
    `return` total amount in €uro
    """
    total = 0

    value = [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    for val, amount in zip(value, money_vec):
        total += val*amount

    return round(total, 2)

def check_for_coins(circles, frame, scaleInPXperMM):
    """Checks if the circle size and color correspond to an euro coin.
    `retrun` list corresponding to circles with the name of the coin (None if not a Euro-Coin)
    """

    coinSizeList = check_circle_sizes(circles, scaleInPXperMM)

    coinList = []

    for coinSize, circle, i in zip(coinSizeList, circles[0], range(len(coinSizeList))):
        oneCirc = []
        for type in coinSize:  # since there can be multiple types for one size
            print(i, ' ', end='')
            if type is None:
                oneCirc.append(None)
                print("None, didn't check")
                continue
            
            if check_coin_color(frame, circle, type): 
                oneCirc.append(type)
            else:
                oneCirc.append(None)
            
        coinList.append(oneCirc)

    # filter Out Nones or decide if two coins are plausable
    
    newCoinList = []
    for coins in coinList:
        didChooseOne = False
        for coin in coins:
            if coin != None:
                newCoinList.append(coin)
                didChooseOne = True
                break
        if didChooseOne == False:
            newCoinList.append(None)

    return newCoinList

def check_coin_color(img, circle, coinType:str):
    """Checks if the cointype matches with the color.
    returns true if its a match"""
    mask,_,_ = cv2.split(img)
    mask = mask * 0
    
    radius = round(circle[2] - 2)  # -2px for radius to only catch the coin

    circ0 = round(circle[0])
    circ1 = round(circle[1])

    if coinType == '1e':  # only check outer part
        mask = cv2.circle(mask, (circ0, circ1), radius, (255), -1)
        mask = cv2.circle(mask, (circ0, circ1), round(circle[2] * 0.72), (0), -1) 
    elif coinType == '2e':
        radius = round(radius * 0.7)  # only check inner area
        mask = cv2.circle(mask, (circ0, circ1), radius, (255), -1)
    else:
        mask = cv2.circle(mask, (circ0, circ1), radius, (255), -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    #cv2.imshow("themask", mask)
    #cv2.waitKey(0)

    b, g, r = cv2.split(img)

    m1 = 1.0/np.sqrt(6) * (2.0*r - g - b)
    m2 = 1.0/np.sqrt(2) * (1.0*g - b)

    meanM1 = cv2.mean(m1, mask=mask)[0]
    meanM2 = cv2.mean(m2, mask=mask)[0]


    meanColor = degrees(atan2(meanM2, meanM1))
    if meanColor < 0: meanColor += 360
    print("MeancolorHUE: ", round(meanColor, 2), coinType)
    
    #meanColor *= 2  # convert to [0-359]

    #cv2.waitKey(0)

    # cv2.imshow(mask)

    coinColorDict = {
        "1c":  'ct_red',
        "2c":  'ct_red',
        "5c":  'ct_red',
        "10c": 'ct_yellow',
        "20c": 'ct_yellow',
        "50c": 'ct_yellow',
        "1e":  'euro_yellow',
        "2e":  'euro_yellow',
    }

    theColor = coinColorDict[coinType]

    def is_in_Huerange(value, setHueValue, hueTolerance):
        upper = setHueValue + hueTolerance
        lower = setHueValue - hueTolerance

        if upper >= 359:  # leaving upper bounds
            upper -= 360

            if (value >= 0 and value <= upper) or (value >= lower and value < 360):
                return True
            
        elif lower < 0:  # leaving lower bounds
            lower += 360

            if (value >= lower and value < 360) or (value >= 0 and value <= upper):
                return True
            
        else:  # normal case
            if value >= lower and value <= upper:
                return True

        return False

    result = is_in_Huerange(meanColor, cfg.coinColor[theColor][0], cfg.coinColor[theColor][1])

    return result

def check_circle_sizes(circles, scaleInPXperMM):
    """Checks if the circle sizes correspond to an euro coin.
    `retrun` list corresponding to circles with the name of the coin (None if not a Euro-Coin)
    e.g. [['1c'], ['20ct', '5ct'], ['2e']]
    """

    coinList = []

    # calc tolerances
    tol = cfg.coinDia['tolerance+-']

    cent1tol  = [cfg.coinDia['1c'] - tol,  cfg.coinDia['1c'] + tol]
    cent2tol  = [cfg.coinDia['2c'] - tol,  cfg.coinDia['2c'] + tol]
    cent5tol  = [cfg.coinDia['5c'] - tol,  cfg.coinDia['5c'] + tol]
    cent10tol = [cfg.coinDia['10c'] - tol, cfg.coinDia['10c'] + tol]
    cent20tol = [cfg.coinDia['20c'] - tol, cfg.coinDia['20c'] + tol]
    cent50tol = [cfg.coinDia['50c'] - tol, cfg.coinDia['50c'] + tol]
    euro1tol  = [cfg.coinDia['1e'] - tol,  cfg.coinDia['1e'] + tol]
    euro2tol  = [cfg.coinDia['2e'] - tol,  cfg.coinDia['2e'] + tol]


    for circle in circles[0]:
        diameter = circle[2] * 2 / scaleInPXperMM  # extract diameter in [mm]

        multiple = []

        if cent1tol[0] < diameter and diameter < cent1tol[1]:
            multiple.append('1c')
        if cent2tol[0] < diameter and diameter < cent2tol[1]:
            multiple.append('2c')
        if cent5tol[0] < diameter and diameter < cent5tol[1]:
            multiple.append('5c')
        if cent10tol[0] < diameter and diameter < cent10tol[1]:
            multiple.append('10c')
        if cent20tol[0] < diameter and diameter < cent20tol[1]:
            multiple.append('20c')
        if cent50tol[0] < diameter and diameter < cent50tol[1]:
            multiple.append('50c')
        if euro1tol[0] < diameter and diameter < euro1tol[1]:
            multiple.append('1e')
        if euro2tol[0] < diameter and diameter < euro2tol[1]:
            multiple.append('2e')

        if not multiple:  # no euro coin is matching the diameter
            multiple.append(None)
        
        coinList.append(multiple)
    
    return coinList
