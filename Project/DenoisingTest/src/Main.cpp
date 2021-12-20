#include "Utils.h"

int main() {

	cv::Mat image = cv::imread("E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\Te-gl_0012.jpg");
	//cv::Mat image = cv::imread("E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\sp_img_gray_noise_white.png");
	cv::Mat result = Utils::GaussianFilter(image, 7, 0.8);
	//cv::Mat result = Utils::AdaptiveMedianFilter(image);
	cv::imshow("Test", result);
	cv::waitKey(0);

	//Utils::WriteMSECSVFile();
	//Utils::WriteNoiseCSVFile();
	//Utils::WriteTimesCSVFile();

	return 0;
}

