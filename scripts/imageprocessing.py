#!/usr/bin/env python

from __future__ import print_function
from hsvhistogram import HSVHistogram, BinaryHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib._cntr import Cntr
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
from GUIconfig import testConfiguration



class ImageAlgorithm():
    """
    Image algorithm class
    Bounded with image processing function and configuration setup/update with ROS
    """
    def __init__(self):  
        # basic image instance
        self.frame = None
        self.objectPos = []         # to store four corner of the rect
        self.objectArea = 0         # to store the object area
        self.hsv_output = None
        self.bgr_output = None
        self.binary_contour = None
        self.path_found = 0
        
        # instructions to ROS
        self.passing_angle = False
        self.turn_left = False
        self.turn_right = False
        self.go_left = False
        self.go_right = False
        
        # initiate classifier attributes
        self.desc = None
        self.le = None
        self.model = None        
        self.modelObtain()
        
        # initiate image processing configuration
        self.upper_red = 255
        self.lower_red = 150
        self.upper_blue = 80
        self.lower_blue = 0
        self.upper_green = 150
        self.lower_green = 20
        self.thr_max_value = 255
        self.thr_block_size = 11
        self.thr_constant = 2
        
        # initiate rectangle orientation
        self.vector_x = []
        self.vector_y = [] 
        self.vector_x_avg = 0
        self.vector_y_avg = 0
        self.average_angle = 0
        self.average_angle_list = [] # to store test data
        self.current_angle = 0       # to store current angle calculated by dir vectores
        self.boxBearing = []         # angle returned by the cv2.minAreaRect
        self.lefty_line = 0          # left side of the line on frame
        self.righty_line = 0         # right side of the line on frame
        self.degree_to_turn = 0      # positive means turn right, negative means left  
        
    def modelObtain(self):    
        # initialize the image descriptor
        self.desc = BinaryHistogram([256])
    
        # target and data
        ### path must be changed to local address, this is a testing address
        data_name = "/home/ka/catkin_ws/src/marker_testing/scripts/dataf.npy"
        target_name = "/home/ka/catkin_ws/src/marker_testing/scripts/targetf.npy"
        data = np.load(data_name,mmap_mode='r')
        target = np.load(target_name,mmap_mode='r')
        # grab the unique target names and encode the labels
        targetNames = np.unique(target)
        self.le = LabelEncoder()
        target = self.le.fit_transform(target) # array    
        # construct the training and testing splits
        (trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
            test_size = 0.3, random_state = 42)
        
        # train the classifier
        self.model = RandomForestClassifier(n_estimators = 25, random_state = 84)
        self.model.fit(trainData, trainTarget) # both array
        
    def toHSV(self):
        # using hsv to detect marker, and create mask
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s.fill(255)
        v.fill(255)
        hsv_image = cv2.merge([h, s, v])
        self.hsv_output = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    def colorBound(self):
        # obtain hsv-bgr image
        self.toHSV()
        
        # set color range
        red_range_lower = np.array([self.lower_blue, self.lower_green, self.lower_red]) # B, G, R
        red_range_upper = np.array([self.upper_blue, self.upper_green, self.upper_red])
        mask = cv2.inRange(self.hsv_output, red_range_lower, red_range_upper)
        cv2.imshow("mask", mask)

        # noise reduction on mask
        # erosion followed by dialation on mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # dialation followed by erosion on mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.binary_contour = mask

        # masking out the marker, and showing the image
        res = cv2.bitwise_and(self.frame, self.frame, mask = mask)  
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        self.bgr_output = res
        
        
        # find biggest contour area and draw rectangle over it
        (cnts, hir) = cv2.findContours(res.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            self.objectArea = cv2.contourArea(cnt)
            box = cv2.minAreaRect(cnt)
            #print(box)
            self.boxBearing = box[-1]
            rect = np.int0(cv2.cv.BoxPoints(box))
            self.objectPos = rect
            
            cv2.drawContours(self.frame, cnt, -1, (255, 255, 0), 5)
            cv2.drawContours(self.frame, [rect], -1, (255, 0, 0), 2)
            
            """
            self.objectPos = [int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])]
            self.objectOri = int(box[-1])
            """
            edges = self.touchedEdges()
            
            # fits a line
            rows,cols = self.frame.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(rect, cv2.cv.CV_DIST_L2,0,0.01,0.01)
            self.vector_x.append(vx)
            self.vector_y.append(vy)
            
            
            # testing purpose
            #print("returned bearing", self.boxBearing)
            #print("rect", rect)
            #print("vx, vy",vx, vy)
            #print("x, y", x, y)
            #print("rows, cols", rows, cols)
            #print("edges", edges)
            
            screen_width, screen_height = self.frame.shape[:2];
            
            if vx != 1. and vy != 1. and vy != 0 and vx != 0:
                # indicating a regular slop over the frame
                self.path_found = 1
                self.lefty_line = int((-x*vy/vx) + y)
                self.righty_line = int(((cols-x)*vy/vx)+y)
                cv2.line(self.frame,(cols-1,self.righty_line),(0,self.lefty_line),(0,255,0),2)
                
                if np.rad2deg(np.arctan(- vx/vy)) > 0:
                    self.current_angle = 90 - np.rad2deg(np.arctan(- vx/vy))
                else:
                    self.current_angle = -(90 + np.rad2deg(np.arctan(- vx/vy)))
                    
            elif vy == 0.:
                self.path_found = 1
                self.lefty_line = int(y)
                self.righty_line = self.lefty_line
                cv2.line(self.frame, (cols-1,self.righty_line),(0,self.lefty_line),(0,255,0),2)
                self.current_angle = 0
                
            elif vy == 1.:
                self.path_found = 1
                self.lefty_line = int(x)
                self.righty_line = self.lefty_line
                cv2.line(self.frame, (self.righty_line,0),(self.lefty_line,rows-1),(0,255,0),2)
                if y < screen_height/ 2 :
                    self.current_angle = -90
                else:
                    self.current_angle = 90
            
            print("current angle",self.current_angle)
            print("edges",edges)
            
            if (edges[0] and edges[2]) or (edges[0] and self.current_angle > -10) or (edges[1] and self.current_angle < 10):
                # when the path is on topleft, botleft, 
                self.passing_angle = False

            elif edges[6] or edges[5] or edges[2] or edges[3] or edges[0] or edges[1]:
                # when the left-right are both touched 
                # or top-bottom both touched
                # or left, right are separately touched
                # which means the path is in the middle of the screen or it is vertical,
                # pass angles-to-turn to James
                self.passing_angle = True 
                print("Passing the angle")
                self.degree_to_turn = - (self.current_angle - 0)
                print("degree to turn:", self.degree_to_turn)

        else:
            self.path_found = 0
            
    
    def touchedEdges(self):
        screen_width, screen_height = self.frame.shape[:2]
        #print(self.objectPos, screen_width, screen_height)
        edges = [0,0,0,0,0,0,0] # [top, bottom, left, right, isTouched, isTopBott, isLeftRight]
        for corner in self.objectPos:
            if corner[1] <= 1 and (not edges[0]):
                edges[0] = 1
            if corner[1] >= screen_width-1 and (not edges[1]):
                edges[1] = 1
            if corner[0] <= 1 and (not edges[2]):
                edges[2] = 1
            if corner[0] >= screen_height-1 and (not edges[3]):
                edges[3] = 1
        
        edges[4] = edges[0] or edges[1] or edges[2] or edges[3]
        edges[5] = edges[0] and edges[1]
        edges[6] = edges[2] and edges[3]
        
        return edges
                
    def adaptiveThreshold(self):
        # using adaptive threshold to detect edges
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, self.thr_max_value, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, self.thr_block_size, self.thr_constant)
        cv2.imshow("threshold", thr)       
            
    def averageOri(self):
        if len(self.vector_x) != 0 and len(self.vector_y)!= 0:
            self.vector_x_avg = np.mean(self.vector_x)
            self.vector_y_avg = np.mean(self.vector_y)
            rad = np.arctan(- self.vector_x_avg/self.vector_y_avg)
            print("orientation found:", np.rad2deg(rad))
            self.average_angle = np.rad2deg(rad)
    
    def plotBearing(self):
        print(len(self.boxBearing))
        plt.plot(self.boxBearing, 'bs')
        plt.plot(self.average_angle_list, 'ro')
        plt.show()
            
    def frameAlgorithm(self):
        # combination of both image processing and classifier
        features = self.desc.describe(self.frame, None)
        
        # predict what type of the image is
        item = self.le.inverse_transform(self.model.predict(features))[0]
        self.colorBound()
        if 'PATH' in item.upper() and len(self.objectPos) != 0:
            return 1
        else:
            return 0
        
    def frameAlgorithmTest(self):
        self.colorBound()
        if len(self.objectPos) != None:
            return 1
        else:
            return 0   
        
def testImage():
    imAI = ImageAlgorithm()
    app = QApplication(sys.argv)
    ex = testConfiguration()
    ex.show()

    test_img = cv2.imread("../testimages/t8.jpg")
    

    # obtain value from configuration
    imAI.upper_red = ex.upper_red.value()
    imAI.lower_red = ex.lower_red.value()
    imAI.upper_blue = ex.upper_blue.value()
    imAI.lower_blue = ex.lower_blue.value()
    imAI.upper_green = ex.upper_green.value()
    imAI.lower_green = ex.lower_green.value()
    imAI.thr_max_value = ex.max_value.value()
    imAI.thr_block_size = ex.threshold_block_size.value()
    imAI.thr_constant = ex.constant_sub.value()
      
    imAI.frame = test_img
    #imAI.adaptiveThreshold()
    imAI.colorBound()
    cv2.imshow("binary", imAI.binary_contour)
    cv2.imshow("frame", imAI.frame)
    #imAI.averageOri()
    
    cv2.waitKey(0)

def testingWebcam():
    imAI = ImageAlgorithm()
    app = QApplication(sys.argv)
    ex = testConfiguration()
    ex.show()
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        # obtain value from configuration
        imAI.upper_red = ex.upper_red.value()
        imAI.lower_red = ex.lower_red.value()
        imAI.upper_blue = ex.upper_blue.value()
        imAI.lower_blue = ex.lower_blue.value()
        imAI.upper_green = ex.upper_green.value()
        imAI.lower_green = ex.lower_green.value()
        imAI.thr_max_value = ex.max_value.value()
        imAI.thr_block_size = ex.threshold_block_size.value()
        imAI.thr_constant = ex.constant_sub.value()
        
        ret, imAI.frame = cap.read()
        if imAI.frame != None:
            imAI.colorBound()
            cv2.imshow("bgr", imAI.hsv_output)
            print(imAI.upper_red, imAI.lower_red, imAI.upper_blue, imAI.lower_blue,\
                   imAI.upper_green, imAI.lower_green)
            """
            if imAI.frameAlgorithm() != 0:
                print("path found")
                print(imAI.objectPos)
            """    
            if cv2.waitKey(1) & 0XFF == ord("q"):
                break
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(app.exec_())

def testingVideo():
    imAI = ImageAlgorithm()
    app = QApplication(sys.argv)
    ex = testConfiguration()
    ex.show()
        
    cap = cv2.VideoCapture("/home/ka/Desktop/new.avi")
    while(cap.isOpened()):
        
        # obtain value from configuration
        imAI.upper_red = ex.upper_red.value()
        imAI.lower_red = ex.lower_red.value()
        imAI.upper_blue = ex.upper_blue.value()
        imAI.lower_blue = ex.lower_blue.value()
        imAI.upper_green = ex.upper_green.value()
        imAI.lower_green = ex.lower_green.value()
        imAI.thr_max_value = ex.max_value.value()
        imAI.thr_block_size = ex.threshold_block_size.value()
        imAI.thr_constant = ex.constant_sub.value()
                
        # obtain video frame, display the hsv image
        ret, imAI.frame = cap.read()
        if imAI.frame != None:
            imAI.frameAlgorithmTest()
            #print(imAI.upper_red, imAI.lower_red, imAI.upper_blue, imAI.lower_blue, imAI.upper_green, imAI.lower_green)

            #if imAI.frameAlgorithm() != 0:
            #    print("path found")
            cv2.imshow("output", imAI.frame)    
            
                       
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break
        #cv2.waitKey(0)
    
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
    
    imAI.averageOri()
    
# ============================ end of class and functions ======================


    
if __name__ == "__main__":
    #testRotation()
    #testingVideo()
    #testingWebcam()
    testImage()
    #testingWebcam()
    #testGUI()
    


