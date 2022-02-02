#pragma once
#include "DenoisingAlgorithms.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <numbers>
#include <chrono>
#include <omp.h>

#ifdef G_LIB_as_DLL
#define ALGORITHMSLIBRARY_API __declspec(dllexport)
#else
#define ALGORITHMSLIBRARY_API __declspec(dllimport)
#endif

extern "C" ALGORITHMSLIBRARY_API void WriteMSECSVFile();

extern "C" ALGORITHMSLIBRARY_API void WriteNoiseCSVFile();

extern "C" ALGORITHMSLIBRARY_API void WriteTimesCSVFile();

extern "C" ALGORITHMSLIBRARY_API cv::Mat GaussianFilter(cv::Mat& initial, const int kernel_size, const double sigma);

extern "C" ALGORITHMSLIBRARY_API cv::Mat AverageFilter(cv::Mat& initial, const int kernel_size);

extern "C" ALGORITHMSLIBRARY_API uchar AdaptiveProcess(cv::Mat& initial, const int row, const int col, int kernel_size, const int maxSize);

extern "C" ALGORITHMSLIBRARY_API cv::Mat AdaptiveMedianFilter(cv::Mat& initial);

extern "C" ALGORITHMSLIBRARY_API cv::Mat BilateralFilter(cv::Mat& initial, const int kernel_size, const double space_sigma, const double color_sigma);

extern "C" ALGORITHMSLIBRARY_API cv::Mat ApplyDenoisingAlgorithm(cv::Mat& img, const int kernel_size, Denoising_Algorithms type);

extern "C" ALGORITHMSLIBRARY_API std::vector<std::string> GetFilePaths(const std::string& path);

extern "C" ALGORITHMSLIBRARY_API double GetMSE(const cv::Mat& initial, const cv::Mat& modified);

extern "C" ALGORITHMSLIBRARY_API std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);

extern "C" ALGORITHMSLIBRARY_API double EstimateNoise(const cv::Mat& img);

extern "C" ALGORITHMSLIBRARY_API std::vector<double> GetSigmaWithFilter(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);

extern "C" ALGORITHMSLIBRARY_API std::chrono::milliseconds GetRunningTime(cv::Mat& img, const int kernel_size, const Denoising_Algorithms& type);

extern "C++" ALGORITHMSLIBRARY_API std::vector<std::chrono::milliseconds> GetAllRunningTimes(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);


/*
static class Utils 
{
public:
	inline static const std::string RELATIVE_PATH = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\";
	static void WriteMSECSVFile();
	static void WriteNoiseCSVFile();
	static void WriteTimesCSVFile();

	static cv::Mat GaussianFilter(cv::Mat& initial, const int kernel_size, const double sigma);
	static cv::Mat AverageFilter(cv::Mat& initial, const int kernel_size);
	static uchar AdaptiveProcess(cv::Mat& initial, const int row, const int col, int kernel_size, const int maxSize);
	static cv::Mat AdaptiveMedianFilter(cv::Mat& initial);
	static cv::Mat BilateralFilter(cv::Mat& initial, const int kernel_size, const double space_sigma, const double color_sigma);

private:
	//Algorithms:

	//Utils
	static cv::Mat ApplyDenoisingAlgorithm(cv::Mat& img, const int kernel_size, Denoising_Algorithms type);
	static std::vector<std::string> GetFilePaths(const std::string& path);
	static double GetMSE(const cv::Mat& initial, const cv::Mat& modified);
	static std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
	static double EstimateNoise(const cv::Mat& img);
	static std::vector<double> GetSigmaWithFilter(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
	static std::chrono::milliseconds GetRunningTime(cv::Mat& img, const int kernel_size, const Denoising_Algorithms& type);
	static std::vector<std::chrono::milliseconds> GetAllRunningTimes(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
};*/
