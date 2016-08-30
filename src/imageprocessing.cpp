/*
 * imageprocessing.coo
 *
 * TODO: display image processing results
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

#include <boost/filesystem.hpp>

//#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "imageprocessing.h"

using namespace aquaursa;
using namespace cv;
using namespace std;

int main( int argc, char** argv){
	if(argc <2){
		cerr << "Usage path_node <video_file> (optional save_folder)" << std::endl;
		return -1;
	}
	cout << argv[1] << std::endl;

	VideoCapture cap(argv[1]);
	if( not cap.isOpened() ){
		std::cerr << "Video file " << argv[1] << "could not be opened." << std::endl;
		return -1;
	}

	//cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;

    namedWindow("TestingVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	aquaursa::imageprocessing imgproc;

    while(1){
    	Mat frame;
    	bool  bSuccess = cap.read(frame);

		if (!bSuccess){ //if not success, break loop
			cout << "Cannot read the frame from video file" << endl;
			break;
		}

        Mat hsv_frame = imgproc.toHSV(frame);
        if (imgproc.colorBound(hsv_frame)){
        	Mat drawn = imgproc.drawResults(frame);
        }
        imshow("frame", frame);

        if(imgproc.passAngle() != 0){
        	cout << "Passing angle to James" << endl;
        }

        if(waitKey(30) == 27){ //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			cout << "esc key is pressed by user" << endl;
			break;
		}

    }

}
