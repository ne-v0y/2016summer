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
    
    # create a smach state machine
    sm = smach.StateMachine(outcomes = ["path_task_done"])
    
    # open the container
    with sm:
        # add states to the container
        smach.StateMachine.add("path finding", path_inScreen(),
                               transitions = {'path found':'path turning', 'path not found':"path finding"})
        smach.StateMachine.add("path turning", path_turned(),
                               transitions = {'turned':'path centralization', 'not turned':'path turning'})
        smach.StateMachine.add("path centralization", path_onMiddle(),
                               transitions = {'centralized':'path_task_done','not centralized':'path centralization'})
    # create and start the introspection server
    sis = smach_ros.IntrospectionServer('path_tasks', sm, '/IMPRROC_BOTTOM')
    sis.start()

    # execute smach plan
    outcome = sm.execute()
    
    rospy.spin()
    sis.stop()
    

    
    
if __name__ == '__main__':
    main()
         