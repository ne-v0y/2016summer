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
        
        #publishers
        self.bottom_pub = rospy.Publisher('bottom_view_binary', Image, queue_size= 1)
        self.bottom_algorithm_pub = rospy.Publisher('path_detection', Image, queue_size = 1)
        #self.hor_baseSpeed_pub = rospy.Publisher('/motor/hor/baseSpeed', MCBaseSpeed, queue_size = 1)        
        #self.ver_baseSpeed_pub = rospy.Publisher('/motor/ver/baseSpeed', MCBaseSpeed, queue_size = 1)
        #self.str_baseSpeed_pub = rospy.Publisher('/motor/str/baseSpeed', MCBaseSpeed, queue_size = 1)
        self.diff_pub = rospy.Publisher('/motor/ver/differential', MCDiff, queue_size = 1)
        self.str_pub = rospy.Publisher('/motor/str/baseSpeed', MCBaseSpeed, queue_size = 1)
        self.my_angle_pub = rospy.Publisher('path_angle', Vector3, queue_size = 1)
        self.my_angle_sub = rospy.Subscriber('/path_angle', Vector3, self.angle_callback, queue_size =1)        
        #subscribers     
        print("connecting...")
        self.bottom_sub = rospy.Subscriber('/bottom/camera/image_raw', Image,self.subBottomCallback, queue_size = 1)
        print("done")
        self.imu_sub = rospy.Subscriber('/os5000/euler',Vector3,self.heading_callback,queue_size=1)
        
        #self.baseSpeed_sub = rospy.Subscriber('/motor/ver/baseSpeed', MCBaseSpeed,self.subBaseSpeedCallback, queue_size = 1)
    
    def angle_callback(self, data):
        print("data from callback",data.x)
            
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
                 
    def webCamTesting(self):
        cap = cv2.VideoCapture(0)
        while(True):
            # capture frame by frame
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            box = self.imAI.objectPos
            if box != 0:
                print("path found", box)
                self.setOrientation(box) 
            if cv2.waitKey(1) & 0XFF == ord("q"):
                break
        
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
        """
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
        """
            
        # turn to heading
        #while(not rospy.is_shutdown() and self.current_heading != goal):
        while(not rospy.is_shutdown()):
            #print("started")
            
            if(self.bottom_img != None):
                print("got bottom image")
                #print("path found, passing my angle to James---------------")
                #self.my_angle_pub.publish(x = 34.0)    
                
                if self.publishBottomAlgorithm():
                    
                    frame_size = self.imAI.frame.shape[:2]
                    height_goal_upper = int(frame_size[0] * 1.3)
                    height_goal_lower = int(frame_size[0] * 0.7)
                    
                    # check if it isTopBott
                    if self.imAI.passing_angle:
                        # touching the left or right, it's fine
                        print("path found, passing my angle to James---------------")
                        self.my_angle_pub.publish(x = self.imAI.current_angle)    
                    
                    # check first if the lefty and righty points are among the bearable intervals
                    elif self.imAI.lefty_line > height_goal_upper and self.imAI.righty_line > height_goal_upper:
                        print("go right")
                        self.str_baseSpeed = 30
                        self.str_pub.publish(baseSpeed = self.str_baseSpeed)
                        
                    elif self.imAI.lefty_line < height_goal_lower and self.imAI.righty_line < height_goal_lower:
                        print("go left")
                        self.str_baseSpeed = -30
                        self.str_pub.publish(baseSpeed = self.str_baseSpeed)
                        
                    elif self.imAI.lefty_line > height_goal_upper and self.imAI.righty_line < height_goal_upper:
                        print("turn right")
                        self.speedDiff = 30
                        self.diff_pub.publish(differential = self.speedDiff)
                        
                    elif self.imAI.lefty_line < height_goal_upper and self.imAI.righty_line > height_goal_lower:
                        print("turn left")
                        self.speedDiff = -30
                        self.diff_pub.publish(differential = self.speedDiff)

                    """    
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
                    """                     
                
                else: 
                    # when there is no path detected
                    self.speedDiff = 0
                    self.diff_pub.publish(differential = self.str_baseSpeed)
                    self.str_baseSpeed = 0
                    self.str_pub.publish(baseSpeed = self.str_baseSpeed)
                    
                    self.imAI.lefty_line = 0
                    self.imAI.righty_line = 0 
                    self.imAI.current_angle = 0
                    self.imAI.path_found = 0
                    
                rate.sleep()
                
            

if __name__ == '__main__':
    bridge = VisionBridge()
    bridge.main(sys.argv)            
        
    

        
