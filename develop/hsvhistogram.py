# import the necessary packages
import numpy as np
import cv2

class HSVHistogram:
    def __init__(self, bins):
        self.bins = bins
        
    def describe(self, image, mask = None):
        # compute the 2D HSV histogram in the HSV color space,
        # then normalize the histogram so that the images with
        # the same content, but either scaled larger or smaller will still
        # have roughly the same histogram
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s.fill(255)
        v.fill(255)
        hsv_image = cv2.merge([h, s, v])
        output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        hist = cv2.calcHist([output], [0, 1, 2], 
            mask, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)
        
        # return out 3D histogram as a flattened array
        return hist.flatten()
    

class BinaryHistogram:
    def __init__(self, bins):
        self.bins = bins
    
    def describe(self, binary, mask = None):
        # compute the histogram of binary path shape images
        # then normalize the histogram
        hist = cv2.calcHist([binary], [0], mask, self.bins, [0, 256])
        hist = cv2.normalize(hist)
        
        return hist.flatten()
    