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

void Utils::ResizeImage(cv::Mat& inputImage, cv::Size size)
{
	cv::Size newSize;

	if (inputImage.empty())
	{
		return;
	}
	if (inputImage.size() == size)
	{
		return;
	}
	else
	{
		double aspectRatio = double(inputImage.size().width) / double(inputImage.size().height);

		double dFactorWidth = double(size.width) / double(inputImage.size().width);
		double dFactorHeight = double(size.height) / double(inputImage.size().height);

		if (dFactorWidth < dFactorHeight)
		{
			double newWidth = double(size.width);
			double newHeight = newWidth / aspectRatio;

			newSize = cv::Size(newWidth, newHeight);
		}
		else if (dFactorWidth > dFactorHeight)
		{
			double newHeight = double(size.height);
			double newWidth = newHeight * aspectRatio;

			newSize = cv::Size(newWidth, newHeight);
		}

		cv::resize(inputImage, inputImage, newSize);
	}
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
