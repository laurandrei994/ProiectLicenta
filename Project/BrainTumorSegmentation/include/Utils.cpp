#include "..\include\Utils.h"
#include <iostream>
#include <fstream>

cv::Mat Utils::ReadImage(const std::string& imagePath)
{
	cv::Mat image;
	image = cv::imread(imagePath);

	if (image.empty())
	{
		std::cout << "Could not read the image: " << imagePath << std::endl;
		image.setTo(0);
		return image;
	}
	return image;
}

void Utils::ShowImage(const cv::Mat& image)
{
	if (image.empty())
	{
		std::cout << "Image is not loaded properly" << std::endl;
		return;
	}
	cv::imshow("Show Image", image);
	cv::waitKey(0);
}

QImage Utils::ConvertMatToQImage(const cv::Mat& source)
{
	cv::Mat cpyImage = source;

	if (cpyImage.channels() == 4)
	{
		QImage result = QImage((uchar*)cpyImage.data, cpyImage.cols, cpyImage.rows, QImage::Format_ARGB32);
		return result;
	}
	if (cpyImage.channels() == 3)
	{
		QImage result = QImage((uchar*)cpyImage.data, cpyImage.cols, cpyImage.rows, QImage::Format_RGB888);
		return result;
	}
	if (cpyImage.channels() == 1)
	{
		QImage result = QImage((uchar*)cpyImage.data, cpyImage.cols, cpyImage.rows, QImage::Format_Indexed8);
		return result;
	}
}
