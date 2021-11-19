#include "Utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <fstream>

cv::Mat ApplyDenoisingAlgorithm(const cv::Mat& img, const int kernel_size, Denoising_Algorithms type);
std::vector<std::string> GetFilePaths(const std::string& path);
double GetMSE(const cv::Mat& initial, const cv::Mat& modified);
std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
void WriteCSVFile(const std::vector<std::string>& files, const std::vector<double>& average, const std::vector<double>& median, const std::vector<double>& gaussian, const std::vector<double>& bilateral);

const std::string RELATIVE_PATH = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\";

int main() {
	cv::Mat image;
	std::vector<std::string> files = GetFilePaths(RELATIVE_PATH);
	std::vector<double> allMSE_Average = GetAllMSE(files, Denoising_Algorithms::AVERAGE, 5);
	std::vector<double> allMSE_Median = GetAllMSE(files, Denoising_Algorithms::MEDIAN, 5);
	std::vector<double> allMSE_Gaussian = GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 5);
	std::vector<double> allMSE_Bilateral = GetAllMSE(files, Denoising_Algorithms::BILATERAL, 5);
	WriteCSVFile(files, allMSE_Average, allMSE_Median, allMSE_Gaussian, allMSE_Gaussian);

	/*for (const std::string& path : files)
		std::cout << path << std::endl;

	std::string path = RELATIVE_PATH + "Te-gl_0010.jpg";
	image = cv::imread(path);
	cv::Mat result = ApplyDenoisingAlgorithm(image, 5, Denoising_Algorithms::AVERAGE);
	cv::imshow("Result image", result);
	cv::waitKey(0);
	cv::destroyWindow("Result image");

	double mse = GetMSE(image, result);
	std::cout << "MSE = " << mse << std::endl;*/

	return 0;
}

cv::Mat ApplyDenoisingAlgorithm(const cv::Mat& img, const int kernel_size, Denoising_Algorithms type) {
	cv::Mat result;
	cv::Size size(kernel_size, kernel_size);
	switch (type)
	{
	case Denoising_Algorithms::GAUSSIAN:
	{
		//std::cout << "Applied gaussian blurring with a kernel size of (" << kernel_size << "," << kernel_size << ")" << std::endl;
		cv::GaussianBlur(img, result, size, 0);
		break;
	}
	case Denoising_Algorithms::MEDIAN:
	{
		//std::cout << "Applied median blurring with a kernel size of (" << kernel_size << "," << kernel_size << ")" << std::endl;
		cv::medianBlur(img, result, kernel_size);
		break;
	}
	case Denoising_Algorithms::AVERAGE:
	{
		//std::cout << "Applied average blurring with a kernel size of (" << kernel_size << "," << kernel_size << ")" << std::endl;
		cv::blur(img, result, size);
		break;
	}
	case Denoising_Algorithms::BILATERAL:
	{
		//std::cout << "Applied bilateral filtering" << std::endl;
		cv::bilateralFilter(img, result, 5, 75, 75);
		break;
	}
	case Denoising_Algorithms::NONE:
	{
		//std::cout << "Image was not modified" << std::endl;
		result = img;
		break;
	}
	default:
		break;
	}
	return result;
}

std::vector<std::string> GetFilePaths(const std::string& path)
{
	const std::filesystem::path dir_path{ path };
	std::vector<std::string> files; 
	for (const auto& file : std::filesystem::directory_iterator(dir_path))
		files.push_back(file.path().u8string());

	return files;
}

double GetMSE(const cv::Mat& initial, const cv::Mat& modified)
{
	cv::Mat s1;
	cv::absdiff(initial, modified, s1);
	s1.convertTo(s1, CV_32F);
	s1 = s1.mul(s1);

	cv::Scalar s= cv::sum(s1);

	double sse = s.val[0] + s.val[1] + s.val[2];
	double mse = sse / (double)(initial.channels() * initial.total());
	return mse;
}

std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
{
	std::vector<double> allMSE;
	for (std::string path : files)
	{
		cv::Mat initial = cv::imread(path);
		cv::Mat modified = ApplyDenoisingAlgorithm(initial, kernel_size, type);
		double mse = GetMSE(initial, modified);
		allMSE.push_back(mse);
	}
	return allMSE;
}

void WriteCSVFile(const std::vector<std::string>& files, const std::vector<double>& average, const std::vector<double>& median, const std::vector<double>& gaussian, const std::vector<double>& bilateral)
{
	std::ofstream csv_file;
	csv_file.open("mse_results.csv");
	csv_file << "Filename, Algorithm, MSE, \n";
	for (int i = 0; i < average.size(); i++)
		csv_file << files.at(i) << ", " << "Average Blurring, " << average.at(i) << ", \n";
	for (int i = 0; i < median.size(); i++)
		csv_file << files.at(i) << ", " << "Median Blurring, " << median.at(i) << ", \n";
	for (int i = 0; i < gaussian.size(); i++)
		csv_file << files.at(i) << ", " << "Gaussian Blurring, " << gaussian.at(i) << ", \n";
	for (int i = 0; i < bilateral.size(); i++)
		csv_file << files.at(i) << ", " << "Bialteral Filtering, " << bilateral.at(i) << ", \n";

	std::cout << "File was written successfully" << std::endl;
	csv_file.close();
}
