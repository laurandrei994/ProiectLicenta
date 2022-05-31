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
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Dynamic Threshold method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_DynamicThreshold(cv::Mat & image);

///\brief											Function that removes the skull from the MRI image using the Adaptive Window method for calculating the threshold
///\param [in] input				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Adaptive Window method
extern "C" ALGORITHMSLIBRARY_API cv::Mat AdaptiveWindow_Threshold(cv::Mat & input);

///\brief											Function that removes the skull from the MRI image by calling the AdaptiveWindow_Threshold function
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Adaptive Window method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_AdaptiveWindow(cv::Mat & image);

///\brief											Function that removes the skull from the MRI image by using a mask based approach and generating different masks to get to the desired result
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the Mask Based Method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_UsingMask(cv::Mat & image);

///\brief											Function that removes the skull from the MRI image by using K-Means clustering
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the skull stripped image after the K-Means clustering method
extern "C" ALGORITHMSLIBRARY_API cv::Mat SkullStripping_KMeans(cv::Mat & image);
///@}

///\name Segmentation Algorithms
/// Functions that are used for the segmentation stage of the application
///@{

///\brief											Function that removes the unwanted pixels from an image using opening method (erode and dilate)
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the image after the opening process
extern "C" ALGORITHMSLIBRARY_API cv::Mat ImageAfterOpening_UsingBinaryMask(cv::Mat & image);

///\brief											Function that selects the tumor area from the image using the K-Means clustering algorithm
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing modified image after aplying the K-means clustering algorithm
extern "C" ALGORITHMSLIBRARY_API cv::Mat KMeansClustering_Brain(cv::Mat & image);

///\brief											Function that applies the Connected Component algorithm on an image to find the largest connected component
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										std::pair<cv::Mat, int> object containing the labeled image on the first place and the index of the largest component component on the second place
extern "C++" ALGORITHMSLIBRARY_API std::pair<cv::Mat, int> ConnectedComponents(cv::Mat & image);

///\brief											Function that extracts the tumor from the image using the infos from the ConnectedComponents function
///\param [in] image				  cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing only the tumor
extern "C" ALGORITHMSLIBRARY_API cv::Mat ExtractTumorArea(cv::Mat & image);

///\brief											Function that applies a contour on the initial image to highlight the tumor
///\param [in] currentImage		cv::Mat object containing the image with the tumor
///\param [in] initialImage		cv::Mat object containing the unmodified image which had been loaded in the project in the start
///\return										cv::Mat object containing final image with the tumor highlighted
extern "C" ALGORITHMSLIBRARY_API cv::Mat ConstructFinalImage(cv::Mat & currentImage, cv::Mat& initialImage);
///@}

///\name Statistical Helper Algorithms
/// Functions that are used for generating running times and comparing results of different algorithms
///@{

///\brief											Function that applies a denoising algorithm on a given image
///\param [in] img						cv::Mat object containing the image that the user wants to modify
///\param [in] kernel_size		constant integer that represents the size of the kernel to be applied on the image
///\param [in[ type						Denoising_Algoithms object representing the chosen type of the algorithm to be apllied 
///\return										cv::Mat object containing the image after applying the denoising algorithm
extern "C" ALGORITHMSLIBRARY_API cv::Mat ApplyDenoisingAlgorithm(cv::Mat & img, const int kernel_size, Denoising_Algorithms type);

///\brief											Function that generates a list of all the file paths of the images used in the project
///\param [in] path						constant string that represents the path where the dataset is saved
///\return										std::vector<string> object containing all the file paths for the images of the dataset
extern "C" ALGORITHMSLIBRARY_API std::vector<std::string> GetFilePaths(const std::string & path);

///\brief											Function that calculates the Mean Squared Error of two images
///\param [in] initial				cv::Mat object containing the image before applying a denoising algorithm
///\param [in] modified				cv::Mat object containing the image after applying a denoising algorithm
///\return										double representing the Mean Squared Error between \a initial and \a modified
extern "C" ALGORITHMSLIBRARY_API double GetMSE(const cv::Mat & initial, const cv::Mat & modified);

///\brief											Function that calculates the Mean Squared Error for all the images in the dataset after applying all the denoising algorithms
///\param [in] files					constant std::vector<string> object with the file paths of all the dataset images
///\param [in] type						Denoising_Algoithms object representing the chosen type of the algorithm to be apllied
///\param [in] kernel_size		constant integer that represents the size of the kernel to be applied on the image
///\return										std::vector<double> object containing all the Mean Squared Error calculated for all the images in the dataset after applying all the denoising algorithms
extern "C" ALGORITHMSLIBRARY_API std::vector<double> GetAllMSE(const std::vector<std::string>&files, const Denoising_Algorithms & type, const int kernel_size);

///\brief											Function that calculates the noise in a given image
///\param [in] img						cv::Mat object containing the image for which the noise will be calculated
///\return										double representing the estimated noise of the image
extern "C" ALGORITHMSLIBRARY_API double EstimateNoise(const cv::Mat & img);

///\brief											Function that generates a list of calculated noise in every image of the dataset after applying all the denoising algorithms
///\param [in] files					constant std::vector<string> object with the file paths of all the dataset images
///\param [in] type						Denoising_Algoithms object representing the chosen type of the algorithm to be apllied
///\param [in] kernel_size		constant integer that represents the size of the kernel to be applied on the image
///\return										std::vector<double> object containing all the calculated noise for every image in the dataset after apllying all the denoising algorithms
extern "C" ALGORITHMSLIBRARY_API std::vector<double> GetSigmaWithFilter(const std::vector<std::string>&files, const Denoising_Algorithms & type, const int kernel_size);

///\brief											Function that calculates the running time of a denoising algorithm
///\param [in] img						cv::Mat object containing an image on which the denoising algorithm is applied
///\param [in] kernel_size		constant integer that represents the size of the kernel to be applied on the image
///\param [in] type						Denoising_Algoithms object representing the chosen type of the algorithm to be apllied
///\return										std::chrono::milliseconds object containing the time (in milliseconds) needed for the chosen denoising algorithm to run 
extern "C" ALGORITHMSLIBRARY_API std::chrono::milliseconds GetRunningTime(cv::Mat & img, const int kernel_size, const Denoising_Algorithms & type);

///\brief											Function that generates a list of calculated running times of all denoising algorithms for every image of the dataset
///\param [in] files					constant std::vector<string> object with the file paths of all the dataset images
///\param [in] type						Denoising_Algoithms object representing the chosen type of the algorithm to be apllied
///\param [in] kernel_size		constant integer that represents the size of the kernel to be applied on the image
///\return										std::vector<std::chrono::milliseconds> object containing all the times (in milliseconds) needed for all the denoising algorithm to run on every image in the dataset
extern "C++" ALGORITHMSLIBRARY_API std::vector<std::chrono::milliseconds> GetAllRunningTimes(const std::vector<std::string>&files, const Denoising_Algorithms & type, const int kernel_size);

///\brief											Function that generates a CSV file with all the Mean Squared Error results calculated using \a GetAllMSE function
extern "C" ALGORITHMSLIBRARY_API void WriteMSECSVFile();

///\brief											Function that generates a CSV file with all the estimated noise calculated using \a GetSigmaWithFilter function
extern "C" ALGORITHMSLIBRARY_API void WriteNoiseCSVFile();

///\brief											Function that generates a CSV file with all the running times calculated using \a GetAllRunningTimes function
extern "C" ALGORITHMSLIBRARY_API void WriteTimesCSVFile();
///@}

///\name Graphical Helper Algorithms
/// Functions that are used for graphical algorithms visualization
///@{

///\brief											Function that calculates the threshold from an image using the cumulative histogram of that image and the triangle method
///\param [in] img						cv::Mat object containing the image for which the threshold will be calculated
///\param [in] histImage			cv::Mat object containing the histogram which is drawn using the \a histogramDisplay function
///\param [in] thresh					uchar representing the initial threshold for calculating the histogram (\a default = 0)
///\return										integer representing the calculated threshold
extern "C" ALGORITHMSLIBRARY_API int extractThresholdFromHistogram(cv::Mat & img, cv::Mat& histImage, uchar thresh = 0);

///\brief											Function that draws the cumulative histogram of the image and highlight the threshold calculated using the \a extractThresholdFromHistogram function
///\param [in] histogram			constant std::vector<int> containing the values of the histogram
///\param [in] startPoint			cv::Point object containing the start point coordinates of the line used for calculating the threshold
///\param [in] endPoint				cv::Point object containing the end point coordinates of the line used for calculating the threshold
///\param [in] thresh					integer representing the threshold value to be used in drawing the histogram
///\return										cv::Mat object with the drawn histogram
extern "C" ALGORITHMSLIBRARY_API cv::Mat histogramDisplay(const std::vector<int> &histogram, const cv::Point &startPoint, const cv::Point &endPoint, int thresh);

///\brief											Function removes the background from a given image
///\param [in] initial				cv::Mat object containing the image that the user wants to modify
///\return										cv::Mat object containing the image with all the background pixels equal to 0
extern "C" ALGORITHMSLIBRARY_API cv::Mat RemoveBackground(cv::Mat & initial);

///\brief											Function that calculates the gradient of a given image
///\param [in] image					cv::Mat object containing the image fow which the gradient will be calculated
///\return										cv::Mat object containing the gradient of the given image
extern "C" ALGORITHMSLIBRARY_API cv::Mat GradientTest(cv::Mat & image);

///\brief											Function extracts the tumor from a labeled image generated by \a Connected Components algorithm
///\param [in] image					cv::Mat object containing the labeled image
///\param [in] indexMaxLabel	constant integer representing the index of the label which will be used as a threshold for extracting the tumor area from the image
///\return										cv::Mat object containing the image with the tumor area extracted
extern "C" ALGORITHMSLIBRARY_API cv::Mat ExtractTumorFromImage(cv::Mat & image, const int indexMaxLabel);
///@}