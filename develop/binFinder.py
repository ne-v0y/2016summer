#!/usr/bin/env python

import numpy as np
import cv2

def toHSV(image):
    # using hsv to detect marker, and create mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s.fill(255)
    v.fill(255)
    hsv_image = cv2.merge([h, s, v])
    output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return output

def greenRange(image):
    lower_blue = np.array([0, 20, 150]) # B, G, R
    upper_blue = np.array([80, 150, 255])
    mask = cv2.inRange(image, lower_blue, upper_blue)
    
    return mask
    

def detectWhite(image):
    hsv_img = toHSV(image)
    result = greenRange(hsv_img)
    cv2.imshow("re", result)
    
def testing():
    image = cv2.imread('u.jpg')
    detectWhite(image)
    cv2.waitKey(0)

if __name__ == "__main__":
    testing()