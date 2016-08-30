#ifndef VISION_BRIDGE_H
#define VISION_BRIDGE_H

#include <stdio.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

class VisionBridge {
	private:
		ros::NodeHandle nh_;
		ros::Publisher pub;

	public:
		VisionBridge( std::string node, int queue_size = 1) {
			pub = nh_.advertise<sensor_msgs::Image>(node, queue_size);

		}

		cv::Mat rosimg2mat(const sensor_msgs::ImageConstPtr& msg) {
			cv_bridge::CvImagePtr cv_ptr;

			try	{
				cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
			}
			catch (cv_bridge::Exception& e)	{
				ROS_ERROR("cv_bridge exception: %s", e.what());
				throw e;
			}

			return cv_ptr->image;
		}

		sensor_msgs::ImagePtr mat2rosimg(cv::Mat& mat) {
			return cv_bridge::CvImage(std_msgs::Header(), "bgr8", mat).toImageMsg();
		}

		void publishImage(cv::Mat& mat) {
			//Publisher should be advertising sensor_msgs::Image
			pub.publish(mat2rosimg(mat));
		}

		void publishImage(cv::Mat& mat, ros::Publisher& pub) {
			//Publisher should be advertising sensor_msgs::Image
			pub.publish(mat2rosimg(mat));
		}

		void publishImage(cv::Mat& mat, std::string node, int queue_size = 1) {
			ros::Publisher pub = nh_.advertise<sensor_msgs::Image>(node, queue_size);
			publishImage(mat, pub);
		}
};

#endif
