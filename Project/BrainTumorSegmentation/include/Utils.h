#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <QtCore/qstring.h>
#include <QtGui/qimage.h>

class Utils
{
public:
	///\brief										String representing the path where the dataset is saved
	inline static std::string dataSetPath = "E:\\Licenta\\ProiectLicenta\\TestData\\";

	///\brief										Function for reading an image
	///\param [in] imagePath		constant string containing the file path for the image which needs to be read
	///returns									cv::Mat object containing the image read from the given path
	static cv::Mat ReadImage(const std::string& imagePath);

	///\brief										Function for showing an image
	///\param [in] image				cv::Mat object containing the image to be shown
	static void ShowImage(const cv::Mat& image);

	///\brief										Function for resizing an image
	///\param [in] inputImage		cv::Mat object containing the image which will be resized
	///\param [in] size					cv::Size object representing the new size for the \a inputImage
	///\param [out] inputImage	cv::Mat object containing the resized image
	static void ResizeImage(cv::Mat& inputImage, cv::Size size);

	///\brief										Function for converting a cv::Mat image into a QImage
	///\param [in] source				constant cv::Mat object containing the image which will be converted
	///returns									QImage object containing the converted image
	static QImage ConvertMatToQImage(const cv::Mat& source);
};