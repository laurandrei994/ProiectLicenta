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

///\name Graphical Algorithms
/// Functions that are used to modify the input image
///@{

///\brief											Function that transforms an imput image into a grayscale image using the average grayscaling method 
///\param [in] image					cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the grayscale image
extern "C" ALGORITHMSLIBRARY_API cv::Mat GrayScale_Average(const cv::Mat & image);

///\brief											Function applies a blur effect on the input image using the Gaussian Filter method
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\param [in] kernel_size		integer that represents the size of the kernel
///\param [in] sigma					double that represents a parameter for calculating Gaussian function
///\return										cv::Mat object containing image with the Gaussian filter applied on it
extern "C" ALGORITHMSLIBRARY_API cv::Mat GaussianFilter(cv::Mat & initial, const int kernel_size, const double sigma);

///\brief											Function applies a blur effect on the input image using an Average Filter method
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\param [in] kernel_size		integer that represents the size of the kernel
///\return										cv::Mat object containing image with the Average filter applied on it
extern "C" ALGORITHMSLIBRARY_API cv::Mat AverageFilter(cv::Mat & initial, const int kernel_size);

///\brief											Helper function for the Adaptive Median Filter function
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\param [in] row						integer that represents a row in the image
///\param [in] col						integer that represents a column in the image
///\param [in] kernel_size		integer that represents the size of the kernel
///\param [in] maxSize				integer that represents the maximum size of the kernel
///\return										uchar that represents the median value of the pixels inside the kernel
extern "C" ALGORITHMSLIBRARY_API uchar AdaptiveProcess(cv::Mat & initial, const int row, const int col, int kernel_size, const int maxSize);

///\brief											Function applies a blur effect on the input image using an Adaptive Median Filter method
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing image with the Adaptive Median Filter applied on it
extern "C" ALGORITHMSLIBRARY_API cv::Mat AdaptiveMedianFilter(cv::Mat & initial);

///\brief											Function applies a blur effect on the input image using a Bilateral Filter method
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\param [in] kernel_size		integer that represents the size of the kernel
///\param [in] space_sigma		double that represent the sigma value of the filter in the coordinate space. If the value is large, it means that the farther the pixels will affect each other, so that the more similar colors in the larger area can get the same color.
///\param [in] color_sigma		double that represents the sigma value of the color space filter. The larger the value of this parameter is, the wider the color in the neighborhood of the pixel will be mixed together, resulting in a larger semi equal color region. (this parameter can be understood as value domain kernel)
///\return										cv::Mat object containing image with the Bilateral Filter applied on it
extern "C" ALGORITHMSLIBRARY_API cv::Mat BilateralFilter(cv::Mat & initial, const int kernel_size, const double space_sigma, const double color_sigma);

///\brief											Function that makes all the background pixels from the input image equal to 0
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing image with all the background pixels made 0
extern "C" ALGORITHMSLIBRARY_API cv::Mat RemoveBackgroundFromImage(cv::Mat & initial);
///@}

///\name Skull Stripping Algorithms
/// Functions that are used for removing the skull from the input image
///@{

///\brief											Function that removes the skull from the MRI image using the threshold obtained from the input image
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Dynamic Threshold method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_DynamicThreshold(cv::Mat & image);

///\brief											Function that removes the skull from the MRI image using the Adaptive Window method for calculating the threshold
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Adaptive Window method
extern "C" ALGORITHMSLIBRARY_API cv::Mat AdaptiveWindow_Threshold(cv::Mat & input);

///\brief											Function that removes the skull from the MRI image by calling the AdaptiveWindow_Threshold function
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Adaptive Window method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_AdaptiveWindow(cv::Mat & image);

///\brief											Function that removes the skull from the MRI image by using a mask based approach and generating different masks to get to the desired result
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Mask Based Method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_UsingMask(cv::Mat & image);

///\brief											Function that removes the skull from the MRI image by using K-Means clustering
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the K-Means clustering method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_KMeans(cv::Mat & image);
///@}

///\name Segmentation Algorithms
/// Functions that are used for the segmentation stage of the application
///@{
// Segmentation Algorithms
extern "C" ALGORITHMSLIBRARY_API cv::Mat ImageAfterOpening_UsingBinaryMask(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat KMeansClustering_Brain(cv::Mat & image);
extern "C++" ALGORITHMSLIBRARY_API std::pair<cv::Mat, int> ConnectedComponents(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat ExtractTumorArea(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat ConstructFinalImage(cv::Mat & currentImage, cv::Mat& initialImage);
///@}

///\name Statistical Helper Algorithms
/// Functions that are used for generating running times and comparing results of different algorithms
///@{
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
///@}

///\name Graphical Helper Algorithms
/// Functions that are used for graphical algorithms visualization
///@{
extern "C" ALGORITHMSLIBRARY_API int extractThresholdFromHistogram(cv::Mat & img, cv::Mat& histImage, uchar thresh = 0);
extern "C" ALGORITHMSLIBRARY_API cv::Mat histogramDisplay(const std::vector<int> &histogram, const cv::Point &startPoint, const cv::Point &endPoint, int thresh);
extern "C" ALGORITHMSLIBRARY_API cv::Mat RemoveBackground(cv::Mat & initial);
extern "C" ALGORITHMSLIBRARY_API cv::Mat GradientTest(cv::Mat & image);
extern "C" ALGORITHMSLIBRARY_API cv::Mat ExtractTumorFromImage(cv::Mat & image, const int indexMaxLabel);
///@}
