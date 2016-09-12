#!/usr/bin/env python

from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from au_core.msg import MCBaseSpeed, MCDiff, MCRaw
from imageprocessing import ImageAlgorithm, testConfiguration
from geometry_msgs.msg import Vector3
from PyQt4.QtGui import *
from PyQt4.QtCore import *


class VisionBridge:
    def __init__(self):
        self.bridge = CvBridge()
        self.imAI = ImageAlgorithm() # initialize algorithm
        self.bottom_img = None
        self.imAI.frame = None
        self.current_heading = 0
        #self.baseSpeed = 0
        #self.hor_baseSpeed = 0
        #self.vert_baseSpeed = 0 # from zero to 100
        #self.str_baseSpeed = 0
        self.speedDiff = 0
        self.str_baseSpeed = 0
        
        rospy.init_node('image_bridge', anonymous = True)
        #self.srv = Server(markerConfig, self.reconfigure)
        
        #publishers
        self.bottom_pub = rospy.Publisher('bottom_view_binary', Image, queue_size= 1)
        self.bottom_algorithm_pub = rospy.Publisher('path_detection', Image, queue_size = 1)
        #self.hor_baseSpeed_pub = rospy.Publisher('/motor/hor/baseSpeed', MCBaseSpeed, queue_size = 1)        
        #self.ver_baseSpeed_pub = rospy.Publisher('/motor/ver/baseSpeed', MCBaseSpeed, queue_size = 1)
        #self.str_baseSpeed_pub = rospy.Publisher('/motor/str/baseSpeed', MCBaseSpeed, queue_size = 1)
        self.diff_pub = rospy.Publisher('/motor/ver/differential', MCDiff, queue_size = 1)
        self.str_pub = rospy.Publisher('/motor/str/baseSpeed', MCBaseSpeed, queue_size = 1)
        
        #subscribers     
        print("connecting...")
        self.bottom_sub = rospy.Subscriber('/bottom/camera/image_raw', Image,self.subBottomCallback, queue_size = 1)
        print("done")
        self.imu_sub = rospy.Subscriber('/os5000/euler',Vector3,self.heading_callback,queue_size=1)
        
        #self.baseSpeed_sub = rospy.Subscriber('/motor/ver/baseSpeed', MCBaseSpeed,self.subBaseSpeedCallback, queue_size = 1)
    
    def reconfigure(self, config, level):
        rospy.loginfo("dynamic_reconfigurure")
        
    def heading_callback(self, data):
        self.current_heading = data.z
        
    def subBottomCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
            self.bottom_img = cv_image
        except CvBridgeError as e:
            print(e)
    
    def subBaseSpeedCallback(self, data):
        self.baseSpeed = data
                
    def publishBottomView(self):
        bottom = self.imAI.binary_contour
        self.bottom_pub.publish(self.bridge.cv2_to_imgmsg(bottom, encoding = 'passthrough'))
        
    def publishBottomAlgorithm(self):
        self.imAI.frame = self.bottom_img
        self.imAI.colorBound()
        """
        if self.imAI.frameAlgorithmTest() != 0:
        #if self.imAI.frameAlgorithm() != 0:
            print("path found")
            #self.setOrientation(self.imAI.objectPos)
        """
        self.bottom_algorithm_pub.publish(self.bridge.cv2_to_imgmsg(self.bottom_img, encoding='bgr8'))
        self.publishBottomView()
        if self.imAI.path_found:
            return 1
        else:
            return 0
        
    def setOrientation(self, box):
        # negative diff turns left
        if box > -75: # which indicates the path is out of capture from upper side
            self.speedDiff = -10
            self.diff_pub.publish(differential = self.speedDiff)
        elif box < -22:
            self.speedDiff = 10
            self.diff_pub.publish(differential = self.speedDiff)
        else:
            self.speedDiff = 0
            self.diff_pub.publish(differential = self.speedDiff)    
        
    def webCamTesting(self):
        cap = cv2.VideoCapture(0)
        #record_file = open("record.txt", 'w+')
        while(True):
            # capture frame by frame
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            box = self.imAI.objectPos
            if box != 0:
                print("path found", box)
                self.setOrientation(box) 
                #self.now_time = rospy.get_rostime()
                #record_file.write('path found at time %i %i' %(self.now_time.secs, self.now_time.nsecs))
                #print('%i %i'%(self.now_time.secs, self.now_time.nsecs))
            if cv2.waitKey(1) & 0XFF == ord("q"):
                break
        #record_file.close()
        
    def testBearing(self, args):
        rate = rospy.Rate(40)
        
        while(not rospy.is_shutdown() and len(self.imAI.vector_x) < 5000):
            if(self.bottom_img != None):
                self.publishBottomAlgorithm()
            rate.sleep()
        self.imAI.averageOri()
        self.imAI.plotBearing()
    
    def main(self, args):
        rate = rospy.Rate(40)
        
        # get the orientation
        while(not rospy.is_shutdown() and len(self.imAI.vector_x) < 400):
            #self.webCamTesting()
            print("start collecting data")
    
            if(self.bottom_img != None):
                print("got bottom image")
                self.publishBottomAlgorithm()
                
            rate.sleep()
            
        print("End of data collection")
        
        self.imAI.averageOri()
        goal = self.imAI.average_angle + self.current_heading
        if goal < 0 :
            goal = goal + 360
        elif goal > 360:
            goal = goal - 360
            
        # turn to heading
        while(not rospy.is_shutdown() and self.current_heading != goal):
            print("starting turning")
            
            if(self.bottom_img != None):
                print("got bottom image")
                if self.publishBottomAlgorithm():
                
                    if self.imAI.current_angle > 0 :
                        print("starting adjustment")
                        # turn right --positive
                        self.str_baseSpeed = 10
                        self.str_pub.publish(baseSpeed = self.str_baseSpeed)
                        
                    else:
                        print("starting adjustment")
                        # turn left --negative
                        self.str_baseSpeed = -10
                        self.str_pub.publish(baseSpeed = self.str_baseSpeed)
                else:
                    self.speedDiff = 0
                    self.diff_pub.publish(baseSpeed = self.str_baseSpeed)
                
            

if __name__ == '__main__':
    bridge = VisionBridge()
    bridge.testBearing(sys.argv)            
        
    

        
