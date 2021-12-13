#pragma once
#include "DenoisingAlgorithms.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <numbers>
#include <chrono>
#include <omp.h>

static class Utils 
{
public:
	inline static const std::string RELATIVE_PATH = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\";
	static void WriteMSECSVFile();
	static void WriteNoiseCSVFile();
	static void WriteTimesCSVFile();

private:
	//Algorithms:
	static cv::Mat AverageFilter(const cv::Mat& initial, const int kernel_size);
	static uchar AdaptiveProcess(const cv::Mat& initial, const int row, const int col, int kernel_size, const int maxSize);
	static cv::Mat AdaptiveMedianFilter(const cv::Mat& initial);
	static cv::Mat GaussianFilter(const cv::Mat& initial, const int kernel_size, const double sigma);
	static cv::Mat BilateralFilter(const cv::Mat& initial, const int kernel_size, const double space_sigma, const double color_sigma);
	//Utils
	static cv::Mat ApplyDenoisingAlgorithm(const cv::Mat& img, const int kernel_size, Denoising_Algorithms type);
	static std::vector<std::string> GetFilePaths(const std::string& path);
	static double GetMSE(const cv::Mat& initial, const cv::Mat& modified);
	static std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
	static double EstimateNoise(const cv::Mat& img);
	static std::vector<double> GetSigmaWithFilter(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
	static std::chrono::microseconds GetRunningTime(const cv::Mat& img, const int kernel_size, const Denoising_Algorithms& type);
	static std::vector<std::chrono::microseconds> GetAllRunningTimes(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
};
