#!/usr/bin/env python

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



class ImageAlgorithm():
    """
    Image algorithm class
    Bounded with image processing function and configuration setup/update with ROS
    """
    def __init__(self):  
        # basic image instance
        self.frame = None
        self.objectPos = []
        self.objectOri = 0
        self.hsv_output = None
        self.bgr_output = None
        self.binary_contour = None
        self.path_found = 0
        
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
        self.average_angle_list = []
        self.current_angle = 0
        self.boxBearing = []
        
        
    def modelObtain(self):    
        # initialize the image descriptor
        self.desc = HSVHistogram([8, 8, 8])
    
        # target and data
        ### path must be changed to local address, this is a testing address
        data_name = "/home/ka/catkin_ws/src/marker_testing/scripts/data_f.npy"
        target_name = "/home/ka/catkin_ws/src/marker_testing/scripts/target_f.npy"
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
        (cnts, _) = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            box = cv2.minAreaRect(cnt)
            self.boxBearing.append(box[-1])
            rect = np.int32(cv2.cv.BoxPoints(box))
            #if (rect[0][0] - rect[1][0]) / (rect[1][1] - rect[2][1]) > 1 and (rect[0][0] - rect[1][0]) / (rect[1][1] - rect[2][1]) < 11:
            #print("box:", box)
            cv2.drawContours(self.frame, cnt, -1, (255, 255, 0), 5)
            cv2.drawContours(self.frame, [rect], -1, (255, 0, 0), 2)
            
            self.objectPos = [int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])]
            self.objectOri = int(box[-1])
            
            # fits a line
            rows,cols = self.frame.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.cv.CV_DIST_L2,0,0.01,0.01)
            self.vector_x.append(vx)
            self.vector_y.append(vy)
            #print("vx, vy",vx, vy)
            
            if vx != 1. and vy != 1. and vy != 0:
                self.path_found = 1
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                cv2.line(self.frame,(cols-1,righty),(0,lefty),(0,255,0),2)
                self.current_angle = np.rad2deg(np.arctan(- vx/vy))
            #print(self.current_angle)
            
            self.average_angle_list.append(self.current_angle)
            
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
        if 'MARK' in item.upper() and len(self.objectPos) != 0:
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

    test_img = cv2.imread("../testimages/t0.jpg")
    

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
    imAI.averageOri()
    
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
            cv2.imshow("output", imAI.frame)
            #print(imAI.upper_red, imAI.lower_red, imAI.upper_blue, imAI.lower_blue, imAI.upper_green, imAI.lower_green)
            """
            if imAI.frameAlgorithm() != 0:
                print("path found")
                print(imAI.objectPos)
            """    
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break
    
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
    
    imAI.averageOri()
    
# ============================ end of class and functions ======================

class testConfiguration(QWidget):
    def __init__(self, parent = None):
        super(testConfiguration, self).__init__(parent)

        # initiate Qt layout        
        layout = QGridLayout()
        self.setLayout(layout)
        self.setWindowTitle("testing")
        
        # initiate labels
        # upper red, lower red
        self.l1 = QLabel("upper red")
        self.l1.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l1)       
        self.l2 = QLabel("lower red")
        self.l2.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l2)
        
        # blue range
        self.l3 = QLabel("upper blue")
        self.l3.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l3)        
        self.l4 = QLabel("lower blue")
        self.l4.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l4)
        
        
        # green range
        self.l5 = QLabel("upper green")
        self.l5.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l5)        
        self.l6 = QLabel("lower green")
        self.l6.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l6)
        
        self.l7 = QLabel("max Value")
        self.l7.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l7, 0, 3)
        self.l8 = QLabel("Threshold block size")
        self.l8.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l8, 1, 3)
        self.l9 = QLabel("Constant")
        self.l9.setAlignment(Qt.AlignLeading)
        layout.addWidget(self.l9, 2, 3)
        
        # initiate sliders
        self.upper_red = self.sliderMaker(0, 255, 255)
        self.lower_red = self.sliderMaker(0, 255, 150)
        self.upper_blue = self.sliderMaker(0, 255, 80)
        self.lower_blue = self.sliderMaker(0, 255, 0)
        self.upper_green = self.sliderMaker(0, 255, 150)
        self.lower_green = self.sliderMaker(0, 255, 20)
        self.max_value = self.sliderMaker(0, 255, 255)
        self.threshold_block_size = self.sliderMaker(3, 27, 11, interval= 2)
        self.constant_sub = self.sliderMaker(0, 100, 2)
        
        # add slider to layout  
        layout.addWidget(self.upper_red, 0, 2)
        layout.addWidget(self.lower_red, 1, 2)
        layout.addWidget(self.upper_blue, 2, 2)
        layout.addWidget(self.lower_blue, 3, 2)
        layout.addWidget(self.upper_green, 4, 2)
        layout.addWidget(self.lower_green, 5, 2)
        layout.addWidget(self.max_value, 0, 5)
        layout.addWidget(self.threshold_block_size, 1, 5)
        layout.addWidget(self.constant_sub, 2, 5)
        
        # slider signals
        self.upper_red.valueChanged.connect(self.changeValue)
        self.lower_red.valueChanged.connect(self.changeValue)
        self.upper_blue.valueChanged.connect(self.changeValue)
        self.lower_blue.valueChanged.connect(self.changeValue)
        self.upper_green.valueChanged.connect(self.changeValue)
        self.lower_green.valueChanged.connect(self.changeValue)
        self.max_value.valueChanged.connect(self.changeValue)
        self.threshold_block_size.valueChanged.connect(self.changeValue)
        self.constant_sub.valueChanged.connect(self.changeValue)
        
        # show values of slider bars
        self.l1_val = QLabel("%d" % self.upper_red.value())
        self.l2_val = QLabel("%d" % self.lower_red.value())
        self.l3_val = QLabel("%d" % self.upper_blue.value())
        self.l4_val = QLabel("%d" % self.lower_blue.value())
        self.l5_val = QLabel("%d" % self.upper_green.value())
        self.l6_val = QLabel("%d" % self.lower_green.value())
        self.l7_val = QLabel("%d" % self.max_value.value())
        self.l8_val = QLabel("%d" % self.threshold_block_size.value())
        self.l9_val = QLabel("%d" % self.constant_sub.value())
        
        # add values to layout
        layout.addWidget(self.l1_val, 0, 1)
        layout.addWidget(self.l2_val, 1, 1)
        layout.addWidget(self.l3_val, 2, 1)
        layout.addWidget(self.l4_val, 3, 1)
        layout.addWidget(self.l5_val, 4, 1)
        layout.addWidget(self.l6_val, 5, 1)
        layout.addWidget(self.l7_val, 0, 4)
        layout.addWidget(self.l8_val, 1, 4)
        layout.addWidget(self.l9_val, 2, 4)
        
    def changeValue(self):
        print("value changed")
        self.l1_val.setText("%d" % self.upper_red.value())
        self.l2_val.setText("%d" % self.lower_red.value())
        self.l3_val.setText("%d" % self.upper_blue.value())
        self.l4_val.setText("%d" % self.lower_blue.value())
        self.l5_val.setText("%d" % self.upper_green.value())
        self.l6_val.setText("%d" % self.lower_green.value())
        self.l7_val.setText("%d" % self.max_value.value())
        self.l8_val.setText("%d" % self.threshold_block_size.value())
        self.l9_val.setText("%d" % self.constant_sub.value())

    def sliderMaker(self, min_val, max_val, set_val, interval = 1):
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(min_val)
        sl.setMaximum(max_val)
        sl.setTickPosition(QSlider.TicksBelow)
        sl.setTickInterval(interval)
        sl.setValue(set_val)
        
        return sl  

def testGUI():
    app = QApplication(sys.argv)
    ex = testConfiguration()
    ex.show()
    sys.exit(app.exec_())
    
    
# ============================ end of class and functions ======================
    
if __name__ == "__main__":
    #testRotation()
    testingVideo()
    #testingWebcam()
    #testImage()
    #testingWebcam()
    #testGUI()
    


