#!/usr/bin/env python

from __future__ import division
import roslib
import rospy
import smach
import smach_ros
from smach_ros import SimpleActionState
from geometry_msgs.msg import Vector3
from au_core.msg import MCBaseSpeed,MCDiff
from std_msgs.msg import Empty
from smach import Sequence


class path_inScreen(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ["path found", "path not found"])
        self.pos_sub_ = rospy.Subscriber('/path_task', Vector3, self.pos_sub_cb, queue_size = 1)
        self.cb_data = Vector3(0,0,0)
    
    def pos_sub_cb(self, data):
        self.cb_data = data
        print(self.cb_data)
    
    def execute(self, userdata):
        rospy.loginfo('Executing state path_inScreen')
        while(self.cb_data != None):
            if self.cb_data.x == 1 and self.cb_data.y == 0 and self.cb_data.z == 0:
                return "path found"
            else:
                return "path not found"
        

class path_turned(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ["turned","not turned"])
        self.pos_sub_ = rospy.Subscriber('/path_task', Vector3, self.pos_sub_cb, queue_size = 1)
        self.cb_data = Vector3(0,0,0)
    
    def pos_sub_cb(self, data):
        self.cb_data = data
        print(self.cb_data)
    
    def execute(self, userdata):
        rospy.loginfo('Executing state path_turned')
        if(self.cb_data != None):
            if self.cb_data.x == 1 and self.cb_data.y == 1 and self.cb_data.z == 0:
                return "turned"
            else:
                return "not turned"
        
        
class path_onMiddle(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes = ["centralized", "not centralized"])   
        self.pos_sub_ = rospy.Subscriber('/path_task', Vector3, self.pos_sub_cb, queue_size = 1)
        self.cb_data = Vector3(1,1,1)
    
    def pos_sub_cb(self, data):
        self.cb_data = data
        print(self.cb_data)
    
    def execute(self, userdata):
        rospy.loginfo('Executing state path_turned')
        while(self.cb_data != None):
            if self.cb_data.x == 1 and self.cb_data.y == 1 and self.cb_data.z == 1:
                return "centralized"
            else:
                return "not centralized"
            
def main():
    rospy.init_node('testing_state_machine');
    
    # create a smach state sequence
    sq = Sequence(outcomes = ['succeeded', 'aborted', 'preempted'],
                  connector_outcome = 'succeeded');
    with sq:
        Sequence.add('finding path', path_inScreen())
        Sequence.add('turning path', path_turned())
        Sequence.add('centralizing path', path_onMiddle);
    
    
if __name__ == '__main__':
    main()
