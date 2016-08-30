/*
 * imageprocessing.h
 *
 * Created on: Jul 16 2016
 * Author    : noni
 *
 */

#ifndef AU_PATH_SRC_IMAGEPROCESSING_H_
#define AU_PATH_SRC_IMAGEPROCESSING_H_

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.cpp>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

namespace aquaursa {

class imageprocessing {

typedef vector<vector<Point> > contours;
typedef vector<Point> single_contour;
typedef vector< Vec4i > hierarchy;

private:
	contours cnts_;
	single_contour minRect;
	//Point2f minRect[4];
	float line_lefty;
	float line_righty;
	float current_angle;
	hierarchy hier_;

public:
	imageprocessing();
	~imageprocessing();

	Mat toHSV( const Mat& frame);
	int colorBound( const Mat& frame);
	Mat drawResults( const Mat& frame);
	Point2f passAngle(void);
	int* touchedEdges();
	Mat enhancement( const Mat& frame, int alpha, int beta);
	Mat findEdge(const Mat& frame);
	Mat newAlgo(const Mat& frame);

	Mat binary;
	Mat tmp;

	int red_lower;
	int red_upper;
	int blue_lower;
	int blue_upper;
	int green_lower;
	int green_upper;

	int cols_;
	int rows_;
	bool image_approved_;
};

imageprocessing::imageprocessing(){

}

imageprocessing::~imageprocessing(){

}

Mat imageprocessing::newAlgo(const Mat& frame){
	Mat binary_;
	Mat canny_o;
	Mat output;
	output = frame;

	//inRange(frame, Scalar(200, 200, 200), Scalar(255, 255, 255), frame);
	cvtColor(frame, binary_, CV_BGR2GRAY);

	int thresh_;
	thresh_ = 100;

	Canny( binary_, canny_o, thresh_, thresh_*2, 3 );

	binary = canny_o;
	dilate(canny_o, canny_o, cv::Mat(), cv::Point(-1,-1));
	erode( canny_o, canny_o, cv::Mat(), cv::Point(-1, -1) );

	vector<Vec2f> lines;
	HoughLines(canny_o, lines, 1, CV_PI/180, 100, 0, 0 );
	//cvtColor(canny_o, output, CV_GRAY2BGR );
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line( output, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	}
	/*
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
	}
	*/


	return output;
}

// @function toHSV
Mat imageprocessing::toHSV( const Mat& frame ){

	//Mat newframe = imageprocessing::enhancement(frame, 2.2, 50 );

	rows_ = frame.rows;									// height
	cols_ = frame.cols;									// width

	Mat frame_hsv;
	Mat channel_hsv[3];
	Mat output;

	cvtColor(frame, frame_hsv, CV_BGR2HSV);

	split(frame_hsv, channel_hsv);	 						// stored as h, s, v
	//channel_hsv[0] = Scalar(255);
	channel_hsv[1] = Scalar(255); // for pool testing
	channel_hsv[2] = Scalar(255);
	merge(channel_hsv, 3, frame_hsv);					// fill the h and v space with 255

	cvtColor(frame_hsv, output, CV_HSV2BGR);
	return output;
}


// @function colorBound
int imageprocessing::colorBound( const Mat& frame){

	Mat mask;
	//Mat mask_eroded;
	//Mat mask_full;

	/*
	red_lower = 150;
	red_upper = 255;
	blue_lower = 0;
	blue_upper = 80;
	green_lower = 20;
	green_upper = 150;
	*/
	//inRange(frame, Scalar(blue_lower,green_lower,red_lower),Scalar(blue_upper,green_upper,red_upper), mask); // B, G, R
	inRange(frame, Scalar(0,0,0), Scalar(1,1,255), mask);
	//cvtColor(mask, binary, CV_BGR2GRAY);
	//binary = mask;
	tmp = mask;


	Mat kernel(Size(5, 5), CV_8UC1);
	kernel.setTo(1);
	morphologyEx(mask, mask, MORPH_OPEN, kernel);  // erodsion
	morphologyEx(mask, mask, MORPH_CLOSE, kernel); // dialation

	Mat result;
	bitwise_and(frame, frame, result, mask);
	cvtColor(result, binary, CV_BGR2GRAY);

	findContours(binary, cnts_, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	if(cnts_.size() > 0){
		return 1;
	}
	else{
		return 0;
	}
}

Mat imageprocessing::findEdge(const Mat& frame){
	Mat binary_;
	Mat canny_o;
	Mat output;
	output = frame;

	//inRange(frame, Scalar(200, 200, 200), Scalar(255, 255, 255), frame);
	cvtColor(frame, binary_, CV_BGR2GRAY);

	int thresh_;
	thresh_ = 80;

	Canny( binary_, canny_o, thresh_, thresh_*2, 3 );

	binary = canny_o;
	dilate(canny_o, canny_o, cv::Mat(), cv::Point(-1,-1));
	//imshow("canny", canny_o);

	/*
	Mat kernel(Size(80, 80), CV_8UC1);
	kernel.setTo(1);
	morphologyEx(canny_o, canny_o, MORPH_OPEN, kernel);  // erodsion
	morphologyEx(canny_o, canny_o, MORPH_CLOSE, kernel); // dialation
	*/
	findContours(canny_o, cnts_, hier_, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	cout<< cnts_.size() <<endl;
	cout << hier_.size() <<endl;

	int i;
	Rect r;
	float area;
	for(i = 0; i < cnts_.size(); i ++){

		drawContours( output, cnts_, i, Scalar(255,0,0), CV_FILLED, 8, hier_, 0, Point());
		area = contourArea(cnts_[i]);
		cout << area << endl;

		Mat approxCurv;
		approxPolyDP(cnts_[i], approxCurv, arcLength(Mat(cnts_[i]), true)*0.02, true);
		cout << approxCurv.size() << endl;

		r = boundingRect(cnts_[i]);
		if(approxCurv.size() != Size(1, 4) ){ //Check if there is a child contour
		//if(sqrt(diff.x*diff.x + diff.y*diff.y) > 2){
			rectangle(output,Point(r.x-10,r.y-10), Point(r.x+r.width+10,r.y+r.height+10), Scalar(0,0,255),2,8,0); //Opened contour
			//putText(output, "fail", Point(r.x-10,r.y-10),FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
		}
		else{
			rectangle(output,Point(r.x-10,r.y-10), Point(r.x+r.width+10,r.y+r.height+10), Scalar(0,255,0),2,8,0); //closed contour
			//putText(output, "i", Point(r.x-10,r.y-10),FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
		}
	}

	return output;

} /* end of findEdge */

// @function drawResults
Mat imageprocessing::drawResults( const Mat& frame ){
	Mat newFrame = frame;

	if(cnts_.size() > 0 ) {
		int index = 0;
		int max = 0;

		for (int i = 0; i < cnts_.size(); ++i) {
			if (contourArea(cnts_[i]) > max) {
				max = contourArea(cnts_[i]);
				index = i;
			}
		}

		single_contour cnt = cnts_[index];
		RotatedRect minRectbox = minAreaRect(cnt);

		// rotated rectangle
		Point2f rect_points[4];
		minRectbox.points( rect_points );
		minRect.resize(4);

		// draw in clockwise, start from the right bottom one
		float distance[4];
		float max_x, max_y, min_y;
		max_x = rect_points[0].x;
		max_y = rect_points[0].y;
		min_y = rect_points[0].y;
		for( int j = 0; j < 4; j++ ){
			line(newFrame, rect_points[j], rect_points[(j+1)%4], Scalar(255, 0, 0), 3, 8 );
			minRect[j] = rect_points[j];
			//Point diff = rect_points[j] - rect_points[(j+1)%4];
			//distance[j] =  sqrt(diff.x*diff.x + diff.y*diff.y);

			if(rect_points[j].x > max_x){
				max_x = rect_points[j].x;
			}
			if(rect_points[j].x > max_y){
				max_y = rect_points[j].y;
			}
			if(rect_points[j].y < min_y){
				min_y = rect_points[j].y;
			}
		}

		if (max_x > cols_/3 && max_y > rows_/3){
			image_approved_ = true;
		}

		drawContours(newFrame, cnts_, index, Scalar(255, 255, 0), 3);

		Vec<float, 4> v_xy;
		fitLine(minRect, v_xy, CV_DIST_L2, 0, 0.01, 0.01);

		//calculate the angle
		float vx = v_xy[0];
		float vy = v_xy[1];
		int x_ = v_xy[2];
		int y_ = v_xy[3];

		//if (distance[0] > distance[3]){

		if(vx != 1. && vy != 1. && vy != 0 && vx != 0){
			line_lefty = (-x_*vy/vx) + y_ ;
			line_righty = ((cols_-x_)*vy/vx)+y_ ;

			float angle = atan( - vx/vy) * 180 / 3.14;
			if (angle > 0){
				current_angle =  90 - angle;
			}
			if (angle < 0){
				current_angle = -(90 + angle);
			}

			Point pt1(cols_-1,line_righty);
			Point pt2(0,line_lefty);
			line(newFrame,pt1, pt2 ,Scalar(0,255,0),2);
		}
		else{
			if(vy == 0.){
				line_lefty = y_;
				line_righty = line_lefty;

				current_angle = 0;

				Point pt1(cols_-1,line_righty);
				Point pt2(0,line_lefty);
				line(newFrame,pt1, pt2 ,Scalar(0,255,0),2);
			}

			if(vy == 1.){
				line_lefty = x_;
				line_righty = y_;

				if (max_x < cols_/2){
					current_angle = 0;
				}
				if (min_y < rows_/2){
					current_angle = -90;
				}
				else{
					current_angle = 90;
				}

				Point pt1(line_righty, 0);
				Point pt2(line_lefty, rows_-1);
				line(newFrame,pt1, pt2 ,Scalar(0,255,0),2);
			}
		}

		cout << current_angle << std::endl;
		//}

	}

	return newFrame;
}


// @function passAngle
Point2f imageprocessing::passAngle(){
	Point2f res(0,0);
	//float res = 0;

	int* edge = imageprocessing::touchedEdges();

	if(edge[4] && current_angle != 0){
		res.x = - (current_angle);
		return res;
	}
	else if(current_angle == 0 || current_angle == -0){
		res.y = line_lefty - rows_/2;  // returning pixels
		return res;
	}

	/*
	if((current_angle == 0 && edge[2]) || (current_angle == 0 && edge[3]) ){
		// only need str motor to move to middle
		return 0;
	}
	else{
		if(edge[4]){
			degree_to_turn = -current_angle;
			return 0;
		}
	}
	*/
}



// @function touchedEdges
int* imageprocessing::touchedEdges(){

	// array stores [top, bottom, left, right, isTouched, isTopBott, isLeftRight]

	int* edge_conditions = new int[7];
	for(int i = 0; i < minRect.size(); i ++){
		if( minRect[i].y <= 1 && !edge_conditions[0]){
			edge_conditions[0] = 1;
		}
		if( ( minRect[i].y >= rows_-1 ) && (!edge_conditions[1])){
			edge_conditions[1] = 1;
		}
		if( ( minRect[i].x <= 1 ) && (!edge_conditions[2])){
			edge_conditions[2] = 1;
		}
		if( ( minRect[i].x >= cols_-1 ) && (!edge_conditions[3])){
			edge_conditions[3] = 1;
		}

		edge_conditions[4] = edge_conditions[0] | edge_conditions[1] | edge_conditions[2] | edge_conditions[3];
		edge_conditions[5] = edge_conditions[0] & edge_conditions[1];
		edge_conditions[6] = edge_conditions[2] & edge_conditions[3];
	}

	return edge_conditions;
}


// @function enhancement
Mat imageprocessing::enhancement( const Mat& frame, int alpha, int beta){
	// alpha [1.0-3.0]: 2.2
	// beta value [0-100]: 50

	Mat newframe = frame;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	for( int y = 0; y < frame.rows; y++ ){
		for( int x = 0; x < frame.cols; x++ ){
			for( int c = 0; c < 3; c++ ){
				newframe.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( frame.at<Vec3b>(y,x)[c] ) + beta );
			 }
		}
	}

	return newframe;
}


} /* namespace aquaursa */


#endif /* AU_PATH_IMAGEPROCESSING_H_ */


