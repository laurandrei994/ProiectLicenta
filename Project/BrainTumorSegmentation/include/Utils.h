#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <QtCore/qstring.h>
#include <QtGui/qimage.h>

class Utils
{
public:
	inline static std::string dataSetPath = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\";
	static cv::Mat ReadImage(const std::string& imagePath);
	static void ShowImage(const cv::Mat& image);
	static void ResizeImage(cv::Mat& inputImage, cv::Size size);
	static QImage ConvertMatToQImage(const cv::Mat& source);
};