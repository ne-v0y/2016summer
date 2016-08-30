/*
 * ros communication node
 * publish path algorithm and pass angle_to_turn to angle_handler
 * to get over on top of the path
 *
 */

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <actionlib/server/simple_action_server.h>
#include <dynamic_reconfigure/server.h>
#include <au_path/path_paraConfig.h>
#include <au_core/MCBaseSpeed.h>


#include "imageprocessing.h"
#include "path_client.h"

//#include "path_ac/path_server.h"

#include <sstream>

static const std::string OPENCV_WINDOW = "Image window";

class connections{

	ros::NodeHandle nh_;

	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	image_transport::Publisher binary_pub_;

	dynamic_reconfigure::Server<au_path::path_paraConfig> config_server_;
	dynamic_reconfigure::Server<au_path::path_paraConfig>::CallbackType configCB_handler_;

	//ros::Publisher distance_pub_;
	//ros::Publisher state_pub_;
	ros::Publisher angle_pub_;

	// path client instance
	//pc::pathclient pclient_;

	// aquaursa class instance
	aquaursa::imageprocessing imgproc;

	int red_lower_;
	int red_upper_;
	int blue_lower_;
	int blue_upper_;
	int green_lower_;
	int green_upper_;

private:
	cv::Mat frame_;


public:
	connections();
	~connections();

	int runAlgorithm();
	void bottomCallback(const sensor_msgs::ImageConstPtr& msg);
	void bottomCallback_new(const sensor_msgs::ImageConstPtr& msg);
	void configCallback(au_path::path_paraConfig &config, uint32_t level);


}; /* class connections */

connections::connections()
	:it_(nh_){

	cout << "initializing nodes" <<std::endl;

	// image subscriber and publisher
	//image_sub_     = it_.subscribe("/bottom/camera/image_raw", 1, &connections::bottomCallback, this);
	image_sub_     = it_.subscribe("/bottom/camera/image_raw", 1, &connections::bottomCallback, this);
	image_pub_     = it_.advertise("vision/path/detect", 1);
	binary_pub_    = it_.advertise("vision/path/binary", 1);
	//distance_pub_  = nh_.advertise<geometry_msgs::Vector3>("/distance_to_central", 1);
	//state_pub_     = nh_.advertise<geometry_msgs::Vector3>("/path_task", 1);
	angle_pub_     = nh_.advertise<geometry_msgs::Vector3>("path_angle", 1);

	configCB_handler_ = boost::bind( &connections::configCallback, this, _1, _2 );
	config_server_.setCallback( configCB_handler_ );

}

connections::~connections(){
	cv::destroyWindow(OPENCV_WINDOW);
}

void connections::bottomCallback_new(const sensor_msgs::ImageConstPtr& msg){
	cv_bridge::CvImagePtr cv_ptr;

	try{
		//cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8") -> image);
		//cv::waitKey(30);
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		frame_ = cv_ptr -> image;
	}
	catch (cv_bridge::Exception& e){
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}

	cv::Mat test;
	test = imgproc.newAlgo(frame_);
	image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", test).toImageMsg());
}

void connections::bottomCallback(const sensor_msgs::ImageConstPtr& msg){

	cv_bridge::CvImagePtr cv_ptr;

	try{
		//cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8") -> image);
		//cv::waitKey(30);
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		frame_ = cv_ptr -> image;
	}
	catch (cv_bridge::Exception& e){
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}

	/*
	imgproc.findEdge(frame_);
	image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame_).toImageMsg());
	binary_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", imgproc.binary).toImageMsg());
	*/

	// start image processing
    cv::Mat hsv = imgproc.toHSV(frame_);

    if (imgproc.colorBound(hsv)){

    	cv::Mat drawn = imgproc.drawResults(frame_);

    	image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", drawn).toImageMsg());
    	binary_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", imgproc.binary).toImageMsg());

    	int angle_count;
    	float angle_sum;
    	angle_sum = 0;
    	if (imgproc.image_approved_ ){
			//cout << "in screen " <<std::endl;

			Point2f angle = imgproc.passAngle();
			cout << angle_sum << std::endl;
			if (angle.x != 0){
				// publish data to path_angle publisher, for state machie
				angle_count = angle_count + 1;
				angle_sum = angle_sum + angle.x;

				if (angle_count > 60){
					cout << "Passing angle to au_path client--------" << std::endl;
					//pclient_.changeHeading(angle.x);
					angle_sum = angle_sum / 60;
					geometry_msgs::Vector3 angle_msg;
					angle_msg.x = angle_sum;
					angle_msg.y = 0;
					angle_msg.z = 0;
					angle_pub_.publish(angle_msg);
				}

			}
			else if (angle.y != 0){

				cout << "moving str(hopefully)" << std::endl;
				//pclient_.changePos(angle.y);

				geometry_msgs::Vector3 angle_msg1;
				angle_msg1.x = angle.y / 100 *30;
				angle_msg1.y = 0;
				angle_msg1.z = 0;
				angle_pub_.publish(angle_msg1);

			}
			else{
				cout << "---------path locked" << std::endl;
			}
    	}


    }
    else{
    	image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame_).toImageMsg());
    	binary_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", imgproc.binary).toImageMsg());
    }

} /* bottom camera callback*/


void connections::configCallback(au_path::path_paraConfig &config, uint32_t level){
	ROS_INFO("Reconfigure request : %i %i %i %i %i %i",
			config.lower_red,
			config.upper_red,
			config.lower_blue,
			config.upper_blue,
			config.lower_green,
			config.upper_green);

	red_lower_ = config.lower_red;
	red_upper_ = config.upper_red;
	blue_lower_ = config.lower_blue;
	blue_upper_ = config.upper_blue;
	green_lower_ = config.lower_green;
	green_upper_ = config.upper_green;

	imgproc.red_lower = red_lower_;
	imgproc.red_upper = red_upper_;
	imgproc.blue_lower = blue_lower_;
	imgproc.blue_upper = blue_upper_;
	imgproc.green_lower = green_lower_;
	imgproc.green_upper = green_upper_;

} /* configuration callback*/


int main(int argc, char** argv){

	ros::init(argc, argv, "path");
	connections con;
	ros::spin();
	return 0;
}
