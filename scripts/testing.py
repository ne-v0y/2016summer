# !/usr/bin/env python

import roslib
import rospy
from geometry_msgs.msg import Vector3
import random

if __name__ == "__main__":
	rospy.init_node("testing", anonymous = True)
	publisher = rospy.Publisher("path_task", Vector3, queue_size = 1)
	rate = rospy.Rate(10)
	while(not rospy.is_shutdown()):
		print("data streaming")

		publisher.publish(Vector3(1, 0, 0))
		rate.sleep()

