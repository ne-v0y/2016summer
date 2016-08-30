/*
 * client of au_pid
 *
 */

#ifndef AU_PATH_SRC_PATH_CLIENT_H_
#define AU_PATH_SRC_PATH_CLIENT_H_

#include <ros/ros.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>

#include <geometry_msgs/Vector3.h>
#include <au_core/Depth.h>
#include <au_core/MCBaseSpeed.h>
#include <au_pid/HeadingControllerAction.h>
#include <path_ac/InMiddleAction.h>
#include <au_core/MCDiff.h>

namespace pc{

class pathclient{
	ros::NodeHandle nh_;
	ros::Subscriber heading_sub_;
	ros::Publisher str_pub_;
	ros::Publisher hor_pub_;

public:
	pathclient();
	~pathclient();

	void headingCallback(const geometry_msgs::Vector3::ConstPtr& msg);
	void headingDoneCb(const actionlib::SimpleClientGoalState& state);
	void movingDoneCb(const actionlib::SimpleClientGoalState& state);
	void changeHeading(float degree_to_test);
	void changePos(float dis_to_move);

	/*
	 * Actionlib clients
	 */
	//actionlib::SimpleActionClient<au_pid::DepthControllerAction> acDepth_;
	actionlib::SimpleActionClient<au_pid::HeadingControllerAction> acHeading_;
	actionlib::SimpleActionClient<path_ac::InMiddleAction> acMiddle_;

private:
	float heading_;

}; /* class pathClient*/


pathclient::pathclient():
	acHeading_( "heading_controller", true),
	acMiddle_( "inMiddle", true){
// client node name should be consistent with the server node name

	acHeading_.waitForServer();
	acMiddle_.waitForServer();

	heading_sub_   = nh_.subscribe("/os5000/euler", 10, &pathclient::headingCallback, this);
	hor_pub_       = nh_.advertise<au_core::MCDiff>("/motor/hor/differential", 1);
	//str_pub_       = nh_.advertise<au_core::MCBaseSpeed>("/motor/str/baseSpeed", 1);
}

pathclient::~pathclient(){

}


void pathclient::headingCallback(const geometry_msgs::Vector3::ConstPtr& msg){
	heading_ = msg->z;
	ROS_INFO("Got heading callback.");
} /* heading callback */

void pathclient::headingDoneCb(const actionlib::SimpleClientGoalState& state){
	 ROS_INFO("Finished in state [%s]", state.toString().c_str());
}/* heading done callback */

void pathclient::movingDoneCb(const actionlib::SimpleClientGoalState& state){
	ROS_INFO("Finished in state [%s]", state.toString().c_str());
} /* path in middle done callback */

void pathclient::changeHeading(float degree_to_turn){

	if((std::abs(degree_to_turn) > 0)){
		au_pid::HeadingControllerGoal goal;

		ROS_INFO( "Heading: %f", heading_);
		//goal.heading = heading_ + rate*pHeading_;
		if (degree_to_turn > 0){
			goal.heading = heading_ - degree_to_turn;
		}
		else{
			goal.heading = heading_ + degree_to_turn;
		}
		ROS_INFO( "Goal heading: %f", goal.heading);

		acHeading_.sendGoal(goal,boost::bind(&pathclient::headingDoneCb, this, _1));
	}
} /* change Heading */

void pathclient::changePos(float dis_to_move){

	path_ac::InMiddleGoal goal_m;
	goal_m.distance_to_go = dis_to_move;

	ROS_INFO("Distance: %f", dis_to_move);
	acMiddle_.sendGoal(goal_m, boost::bind(&pathclient::movingDoneCb, this, _1));

	std::cout << heading_ << std::endl;
	au_pid::HeadingControllerGoal goal;
	goal.heading = heading_;
	acHeading_.sendGoal(goal);

} /* change position*/


} /*namespace pc*/

#endif /* PATH_CLIENT_H_ */

