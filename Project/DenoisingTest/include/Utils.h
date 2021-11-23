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

static class Utils 
{
public:
	inline static const std::string RELATIVE_PATH = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\";
	static void WriteMSECSVFile();
	static void WriteNoiseCSVFile();

private:
	static cv::Mat ApplyDenoisingAlgorithm(const cv::Mat& img, const int kernel_size, Denoising_Algorithms type);
	static std::vector<std::string> GetFilePaths(const std::string& path);
	static double GetMSE(const cv::Mat& initial, const cv::Mat& modified);
	static std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
	static double EstimateNoise(const cv::Mat& img);
	static std::vector<double> GetSigmaWithFilter(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size);
};
