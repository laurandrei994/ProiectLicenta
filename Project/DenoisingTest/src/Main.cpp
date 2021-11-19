#include <iostream>
#include <opencv2/opencv.hpp>
#include "Utils.h"


int main() {
	cv::Mat image;
	image = cv::imread("C:\\Users\\Laur\\Desktop\\New folder\\test_image.png");
	cv::imshow("Test image", image);
	cv::waitKey(0);
	cv::destroyWindow("Test image");

	return 0;
}