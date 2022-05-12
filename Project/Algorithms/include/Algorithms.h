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

//Graphical Algorithms
// GrayScale Algorithm
extern "C" ALGORITHMSLIBRARY_API cv::Mat GrayScale_Average(const cv::Mat & image);

// Denoising Filters
extern "C" ALGORITHMSLIBRARY_API cv::Mat GaussianFilter(cv::Mat & initial, const int kernel_size, const double sigma);
extern "C" ALGORITHMSLIBRARY_API cv::Mat AverageFilter(cv::Mat & initial, const int kernel_size);
extern "C" ALGORITHMSLIBRARY_API uchar AdaptiveProcess(cv::Mat & initial, const int row, const int col, int kernel_size, const int maxSize);
extern "C" ALGORITHMSLIBRARY_API cv::Mat AdaptiveMedianFilter(cv::Mat & initial);
extern "C" ALGORITHMSLIBRARY_API cv::Mat BilateralFilter(cv::Mat & initial, const int kernel_size, const double space_sigma, const double color_sigma);

// Skull Stripping Algorithms
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_DynamicThreshold(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat AdaptiveWindow_Threshold(cv::Mat & input);
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_AdaptiveWindow(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_UsingMask(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_KMeans(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat GradientTest(cv::Mat & image);

// Segmentation Algorithms
extern "C" ALGORITHMSLIBRARY_API cv::Mat ImageAfterOpening_UsingBinaryMask(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat KMeansClustering_Brain(cv::Mat & image);
extern "C++" ALGORITHMSLIBRARY_API std::pair<cv::Mat, int> ConnectedComponents(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat ExtractTumorArea(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat ConstructFinalImage(cv::Mat & currentImage, cv::Mat& initialImage);



// Helper Algorithms
extern "C" ALGORITHMSLIBRARY_API cv::Mat ApplyDenoisingAlgorithm(cv::Mat & img, const int kernel_size, Denoising_Algorithms type);
extern "C" ALGORITHMSLIBRARY_API std::vector<std::string> GetFilePaths(const std::string & path);
extern "C" ALGORITHMSLIBRARY_API double GetMSE(const cv::Mat & initial, const cv::Mat & modified);
extern "C" ALGORITHMSLIBRARY_API std::vector<double> GetAllMSE(const std::vector<std::string>&files, const Denoising_Algorithms & type, const int kernel_size);
extern "C" ALGORITHMSLIBRARY_API double EstimateNoise(const cv::Mat & img);
extern "C" ALGORITHMSLIBRARY_API std::vector<double> GetSigmaWithFilter(const std::vector<std::string>&files, const Denoising_Algorithms & type, const int kernel_size);
extern "C" ALGORITHMSLIBRARY_API std::chrono::milliseconds GetRunningTime(cv::Mat & img, const int kernel_size, const Denoising_Algorithms & type);
extern "C++" ALGORITHMSLIBRARY_API std::vector<std::chrono::milliseconds> GetAllRunningTimes(const std::vector<std::string>&files, const Denoising_Algorithms & type, const int kernel_size);
extern "C" ALGORITHMSLIBRARY_API void WriteMSECSVFile();
extern "C" ALGORITHMSLIBRARY_API void WriteNoiseCSVFile();
extern "C" ALGORITHMSLIBRARY_API void WriteTimesCSVFile();

extern "C" ALGORITHMSLIBRARY_API int extractThresholdFromHistogram(cv::Mat & img, cv::Mat& histImage, uchar thresh = 0);
extern "C" ALGORITHMSLIBRARY_API cv::Mat histogramDisplay(const std::vector<int> &histogram, const cv::Point &startPoint, const cv::Point &endPoint, int thresh);
extern "C" ALGORITHMSLIBRARY_API cv::Mat RemoveBackground(cv::Mat & initial);
extern "C" ALGORITHMSLIBRARY_API cv::Mat RemoveBackgroundFromImage(cv::Mat & initial);
extern "C" ALGORITHMSLIBRARY_API cv::Mat ExtractTumorFromImage(cv::Mat & image, const int indexMaxLabel);
