#include "Algorithms.h"
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <numbers>
#include <chrono>
#include <omp.h>

static std::string RELATIVE_PATH = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\";

ALGORITHMSLIBRARY_API cv::Mat GrayScale_Average(const cv::Mat& image)
{
	cv::Mat cpyImage = image;
	cv::Mat grayMat = cv::Mat(cpyImage.rows, cpyImage.cols, CV_8UC1);

	for (int row = 0; row < image.rows; ++row)
	{
		uchar* cpyImageRow = cpyImage.ptr<uchar>(row);
		uchar* grayMatRow = grayMat.ptr<uchar>(row);

		for (int col = 0; col < image.cols; ++col)
		{
			int sum = 0;
			int pixel = col * cpyImage.channels();
			for (int i = 0; i < cpyImage.channels(); ++i)
			{
				sum += cpyImageRow[pixel + i];
			}
			grayMatRow[col] = round(sum / cpyImage.channels());
		}
	}
	return grayMat;
}

ALGORITHMSLIBRARY_API cv::Mat AverageFilter(cv::Mat& initial, const int kernel_size)
{
	cv::Mat result;
	cv::Point anchorPoint = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);

	cv::filter2D(initial, result, ddepth, kernel, anchorPoint, delta, cv::BORDER_DEFAULT);
	return result;
}

ALGORITHMSLIBRARY_API uchar AdaptiveProcess(cv::Mat& initial, const int row, const int col, int kernel_size, const int maxSize)
{
	std::vector<uchar> pixels;
	for (int a = -kernel_size / 2; a <= kernel_size / 2; a++) 
	{
		uchar* row_ptr = initial.ptr<uchar>(row + a);
		for (int b = -kernel_size / 2; b <= kernel_size / 2; b++)
		{
			pixels.push_back(row_ptr[col + b]);
			//pixels.push_back(initial.at<uchar>(row + a, col + b);
		}
	}
	std::sort(pixels.begin(), pixels.end());
	auto min = pixels[0];
	auto max = pixels[kernel_size * kernel_size - 1];
	auto med = pixels[kernel_size * kernel_size / 2];
	uchar* zxy = initial.ptr<uchar>(row);
	if (med > min && med < max)
	{
		if (zxy[col] > min && zxy[col] < max)
		{
			return zxy[col];
		}
		else
		{
			return med;
		}
	}
	else
	{
		kernel_size += 2;
		if (kernel_size <= maxSize)
			return AdaptiveProcess(initial, row, col, kernel_size, maxSize);
		else
			return med;
	}
}

ALGORITHMSLIBRARY_API cv::Mat AdaptiveMedianFilter(cv::Mat& initial)
{
omp_set_num_threads(8);
	cv::Mat result;
	int minSize = 3; 
	int maxSize = 7;
	cv::copyMakeBorder(initial, result, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
	int rows = result.rows;
	int cols = result.cols;
#pragma omp parallel for
	for (int j = maxSize / 2; j < rows - maxSize / 2; j++)
	{
		uchar* ptr = result.ptr<uchar>(j);
		for (int i = maxSize / 2; i < cols * result.channels() - maxSize / 2; i++)
		{
			ptr[i] = AdaptiveProcess(result, j, i, minSize, maxSize);
		}
	}
	return result;
}

ALGORITHMSLIBRARY_API cv::Mat GaussianFilter(cv::Mat& initial, const int kernel_size, const double sigma)
{
	CV_Assert(initial.channels() == 1 || initial.channels() == 3);
	cv::Mat result = initial.clone();
	double* matrix = new double[kernel_size];
	double sum = 0;
	int origin = kernel_size / 2;
	for (int i = 0; i < kernel_size; i++)
	{
		double g = std::exp(-(i - origin) * (i - origin) / (2 * sigma * sigma));
		sum += g;
		matrix[i] = g;
	}

	//Normalizare
	for (int i = 0; i < kernel_size; i++)
		matrix[i] /= sum;

	//Apply border to image
	int border = kernel_size / 2;
	cv::copyMakeBorder(initial, result, border, border, border, border, cv::BorderTypes::BORDER_REFLECT);
	int channels = result.channels();
	int rows = result.rows - border;
	int cols = result.cols - border;

	//Orizontal
	for (int i = border; i < rows; i++)
	{
		uchar* row_ptr = result.ptr<uchar>(i);
		cv::Vec3b* row_ptr_vec = result.ptr<cv::Vec3b>(i);

		for (int j = border; j < cols; j++)
		{
			std::vector<double> sum(3, 0);
			for (int k = -border; k <= border; k++)
			{
				if (channels == 1)
				{
					sum[0] += matrix[border + k] * row_ptr[j + k];
				}
				else if (channels == 3)
				{
					cv::Vec3b rgb = row_ptr_vec[j + k];
					sum[0] += matrix[border + k] * rgb[0];
					sum[1] += matrix[border + k] * rgb[1];
					sum[2] += matrix[border + k] * rgb[2];
				}
			}
			for (int k = 0; k < channels; k++)
			{
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (channels == 1)
				row_ptr[j] = static_cast<uchar>(sum[0]);
			else if (channels == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				row_ptr_vec[j] = rgb;
			}
		}
	}

	//Vertical
	for (int i = border; i < rows; i++)
	{
		uchar* ptr = result.ptr<uchar>(i);
		cv::Vec3b* ptr_vec = result.ptr<cv::Vec3b>(i);
		for (int j = border; j < cols; j++)
		{
			double sum[3] = { 0 };
			for (int k = -border; k < border; k++)
			{
				uchar* row_ptr = result.ptr<uchar>(i + k);
				cv::Vec3b* row_ptr_vec = result.ptr<cv::Vec3b>(i + k);
				if (channels == 1)
				{
					sum[0] += matrix[border + k] * row_ptr[j];
				}
				else if (channels == 3)
				{
					cv::Vec3b rgb = row_ptr_vec[j];
					sum[0] += matrix[border + k] * rgb[0];
					sum[1] += matrix[border + k] * rgb[1];
					sum[2] += matrix[border + k] * rgb[2];
				}
			}
			for (int k = 0; k < channels; k++)
			{
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (channels == 1)
				ptr[j] = static_cast<uchar>(sum[0]);
			else if (channels == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				ptr_vec[j] = rgb;
			}
		}
	}
	delete[] matrix;
	return result;
}

ALGORITHMSLIBRARY_API cv::Mat BilateralFilter(cv::Mat& initial, const int kernel_size, const double space_sigma, const double color_sigma)
{
	cv::Mat result = initial.clone();
	int channels = initial.channels();
	CV_Assert(channels == 1 || channels == 3);
	double space_coefficient = -0.5 / (space_sigma * space_sigma);
	double color_coefficient = -0.5 / (color_sigma * color_sigma);
	int radius = kernel_size / 2;
	cv::Mat temp;
	cv::copyMakeBorder(initial, temp, radius, radius, radius, radius, cv::BorderTypes::BORDER_REFLECT);
	std::vector<double> _color_weight(channels * 256);
	std::vector<double> _space_weight(kernel_size * kernel_size);
	std::vector<int> _space_ofs(kernel_size * kernel_size);
	double* color_weight = &_color_weight[0];
	double* space_weight = &_space_weight[0];
	int* space_ofs = &_space_ofs[0];

	for (int i = 0; i < channels * 256; i++)
		color_weight[i] = std::exp(i * i * color_coefficient);
	//Generate Space Template
	int maxk = 0;
	for (int i = -radius; i < radius; i++)
	{
		for (int j = -radius; j < radius; j++)
		{
			double r = sqrt(i * i + j * j);
			if (r > radius)
				continue;
			space_weight[maxk] = std::exp(r * r * space_coefficient);
			space_ofs[maxk++] = i * temp.step + j * channels;
		}
	}

	//Filtering process
	for (int i = 0; i < initial.rows; i++)
	{
		const uchar* sptr = temp.data + (i + radius) * temp.step + radius * channels;
		uchar* dptr = result.data + i * result.step;
		if (channels == 1)
		{
			for (int j = 0; j < initial.cols; j++)
			{
				double sum = 0, wsum = 0;
				int val0 = sptr[j]; //Pixeli din centrul template-ului
				for (int k = 0; k < maxk; k++)
				{
					int val = sptr[j + space_ofs[k]];
					double w = space_weight[k] * color_weight[abs(val - val0)]; //Template coefficient = space coefficient * gray value coefficient
					sum += val * w;
					wsum += w;
				}
				dptr[j] = (uchar)cvRound(sum / wsum);
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < initial.cols * 3; j += 3)
			{
				double sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int b0 = sptr[j];
				int g0 = sptr[j + 1];
				int r0 = sptr[j + 2];
				for (int k = 0; k < maxk; k++)
				{
					const uchar* sptr_k = sptr + j + space_ofs[k];
					int b = sptr_k[0];
					int g = sptr_k[1];
					int r = sptr_k[2];
					double w = space_weight[k] * color_weight[abs(b - b0) + abs(g - g0) + abs(r - r0)];
					sum_b += b * w;
					sum_g += g * w;
					sum_r += r * w;
					wsum += w;
				}
				wsum = 1.0f / wsum;
				b0 = cvRound(sum_b * wsum);
				g0 = cvRound(sum_g * wsum);
				r0 = cvRound(sum_r * wsum);
				dptr[j] = (uchar)b0;
				dptr[j + 1] = (uchar)g0;
				dptr[j + 2] = (uchar)r0;
			}
		}
	}
	return result;
}

ALGORITHMSLIBRARY_API cv::Mat RemoveBackground(cv::Mat& initial)
{
	cv::Mat result;
	initial.copyTo(result);
	for (int row = 0; row < initial.rows; ++row)
	{
		const uchar* initialRowPtr = initial.ptr<uchar>(row);
		uchar* resultRowPtr = result.ptr<uchar>(row);

		for (int col = 0; col < initial.cols; ++col)
		{
			if (initialRowPtr[col] < 20)
				resultRowPtr[col] = 0;
		}
	}
	return result;
}

ALGORITHMSLIBRARY_API cv::Mat RemoveBackgroundFromImage(cv::Mat& initial)
{
	cv::Mat hist;

	cv::Mat otsuImg(initial.rows, initial.cols, CV_8UC1);
	double thresh = 0;
	double maxVal = 255;

	cv::threshold(initial, otsuImg, thresh, maxVal, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Mat imgAfterMask;

	cv::dilate(otsuImg, otsuImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25)));
	cv::erode(otsuImg, otsuImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25)));

	cv::bitwise_and(initial, otsuImg, imgAfterMask);

	int thresh2 = extractThresholdFromHistogram(imgAfterMask, hist, 1);

	cv::Mat binaryMask = cv::Mat(initial.rows, initial.cols, CV_8UC1);
	for (int row = 0; row < initial.rows; ++row)
	{
		uchar* imageRow = initial.ptr<uchar>(row);
		uchar* binaryImgRow = binaryMask.ptr<uchar>(row);

		for (int col = 0; col < initial.cols; ++col)
		{
			if (imageRow[col] >= thresh2)
				binaryImgRow[col] = 255;
			else
				binaryImgRow[col] = 0;
		}
	}

	return imgAfterMask;
}

ALGORITHMSLIBRARY_API cv::Mat ApplyDenoisingAlgorithm(cv::Mat& img, const int kernel_size, Denoising_Algorithms type)
{
	cv::Mat result;
	//cv::Size size(kernel_size, kernel_size);
	switch (type)
	{
	case Denoising_Algorithms::GAUSSIAN:
	{
		result = GaussianFilter(img, kernel_size, 0.8);
		break;
	}
	case Denoising_Algorithms::MEDIAN:
	{
		result = AdaptiveMedianFilter(img);
		break;
	}
	case Denoising_Algorithms::AVERAGE:
	{
		result = AverageFilter(img, kernel_size);
		break;
	}
	case Denoising_Algorithms::BILATERAL:
	{
		result = BilateralFilter(img, kernel_size, 75, 75);
		break;
	}
	case Denoising_Algorithms::NONE:
	{
		result = img.clone();
		break;
	}
	default:
		break;
	}
	return result;
}

ALGORITHMSLIBRARY_API std::vector<std::string> GetFilePaths(const std::string& path)
{
	const std::filesystem::path dir_path{ path };
	std::vector<std::string> files;
	for (const auto& file : std::filesystem::directory_iterator(dir_path))
	{
		std::string filepath = file.path().generic_string();
		files.push_back(filepath);
	}
	return files;
}

ALGORITHMSLIBRARY_API double GetMSE(const cv::Mat& initial, const cv::Mat& modified)
{
	cv::Mat s1;
	cv::absdiff(initial, modified, s1);
	s1.convertTo(s1, CV_32F);
	s1 = s1.mul(s1);

	cv::Scalar s = cv::sum(s1);

	double sse = s.val[0] + s.val[1] + s.val[2];
	double mse = sse / (double)(initial.channels() * initial.total());
	return mse;
}

ALGORITHMSLIBRARY_API std::vector<double> GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
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

ALGORITHMSLIBRARY_API double EstimateNoise(const cv::Mat& img)
{
	cv::Mat greyImg;
	cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);
	int height = greyImg.rows;
	int width = greyImg.cols;

	cv::Mat transformMat = (cv::Mat_<int>(3, 3) << 
																1, -2, 1,
																-2, 4, -2,
																1, -2, 1);
	cv::filter2D(greyImg, greyImg, -1, transformMat);

	cv::Scalar sigma = cv::sum(cv::sum(cv::abs(greyImg)));
	//CV_PI
	sigma = sigma * sqrt(0.5 * CV_PI) / (6 * (width - 2) * (height - 1));

	return sigma[0];
}

ALGORITHMSLIBRARY_API std::vector<double> GetSigmaWithFilter(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
{
	int index = 0;
	std::vector<double> results;
	for (std::string path : files)
	{
		cv::Mat initial = cv::imread(path);
		cv::Mat modified = ApplyDenoisingAlgorithm(initial, kernel_size, type);
		double sigma = EstimateNoise(modified);
		std::cout << index++ << std::endl;
		results.push_back(sigma);
	}
	return results;
}

ALGORITHMSLIBRARY_API std::chrono::milliseconds GetRunningTime(cv::Mat& img, const int kernel_size, const Denoising_Algorithms& type)
{
	using Duration = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat modified = ApplyDenoisingAlgorithm(img, kernel_size, type);
	auto stop = std::chrono::high_resolution_clock::now();
	const Duration duration = std::chrono::duration_cast<Duration>(stop - start);

	return duration;
}

ALGORITHMSLIBRARY_API std::vector<std::chrono::milliseconds> GetAllRunningTimes(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
{
	using Duration = std::chrono::milliseconds;
	int index = 0;
	std::vector<Duration> results;
	for (std::string path : files)
	{
		cv::Mat initial = cv::imread(path);
		const Duration duration = GetRunningTime(initial, kernel_size, type);
		results.push_back(duration);
		std::cout << index++ << std::endl;
	}
	return results;
}

ALGORITHMSLIBRARY_API void WriteMSECSVFile()
{
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = GetFilePaths(RELATIVE_PATH);

	std::cout << "Getting the MSE vector using Average Blurring algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Average_3 = GetAllMSE(files, Denoising_Algorithms::AVERAGE, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Average_5 = GetAllMSE(files, Denoising_Algorithms::AVERAGE, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Average_7 = GetAllMSE(files, Denoising_Algorithms::AVERAGE, 7);

	std::cout << "Getting the MSE vector using Median Blurring algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Median_3 = GetAllMSE(files, Denoising_Algorithms::MEDIAN, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Median_5 = GetAllMSE(files, Denoising_Algorithms::MEDIAN, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Median_7 = GetAllMSE(files, Denoising_Algorithms::MEDIAN, 7);

	std::cout << "Getting the MSE vector using Gaussian Blurring algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Gaussian_3 = GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Gaussian_5 = GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Gaussian_7 = GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 7);

	std::cout << "Getting the MSE vector using Bilateral Filtering algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Bilateral_3 = GetAllMSE(files, Denoising_Algorithms::BILATERAL, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Bilateral_5 = GetAllMSE(files, Denoising_Algorithms::BILATERAL, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Bilateral_7 = GetAllMSE(files, Denoising_Algorithms::BILATERAL, 7);

	std::cout << "Started writing the file with the MSE results.. " << std::endl;
	std::ofstream csv_file;
	csv_file.open("mse_results.csv");
	csv_file << "Filename, Average3, Average5, Average7, Median3, Median5, Median7, Gaussian3, Gaussian5, Gaussian7, Bilateral3, Bilateral5, Bilateral7, \n";
	
	std::cout << "\tWriting the results.." << std::endl;
	for (int i = 0; i < files.size(); i++)
	{
		csv_file << files.at(i) << ", " << allMSE_Average_3.at(i) << ", " << allMSE_Average_5.at(i) << ", " << allMSE_Average_7.at(i) <<
			", " << allMSE_Median_3.at(i) << ", " << allMSE_Median_5.at(i) << ", " << allMSE_Median_7.at(i) <<
			", " << allMSE_Gaussian_3.at(i) << ", " << allMSE_Gaussian_5.at(i) << ", " << allMSE_Gaussian_7.at(i) <<
			", " << allMSE_Bilateral_3.at(i) << ", " << allMSE_Bilateral_5.at(i) << ", " << allMSE_Bilateral_7.at(i) << ", \n";
	}

	std::cout << "File was written successfully" << std::endl << std::endl;
	csv_file.close();
}

ALGORITHMSLIBRARY_API void WriteNoiseCSVFile()
{
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = GetFilePaths(RELATIVE_PATH);

	std::cout << "Calculating sigma for inital images..." << std::endl;
	std::vector<double> initial; 
	for (std::string path : files) {
		std::cout << "Calculating" << std::endl;
		cv::Mat img = cv::imread(path);
		double sigma = EstimateNoise(img);
		initial.push_back(sigma);
	}
	
	std::cout << "Calculating sigma vector for images modified with Average Blurring algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<double> average_3 = GetSigmaWithFilter(files, Denoising_Algorithms::AVERAGE, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<double> average_5 = GetSigmaWithFilter(files, Denoising_Algorithms::AVERAGE, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<double> average_7 = GetSigmaWithFilter(files, Denoising_Algorithms::AVERAGE, 7);

	std::cout << "Calculating sigma vector for images modified with Median Blurring algorithm..." << std::endl;
	std::vector<double> adaptive_median = GetSigmaWithFilter(files, Denoising_Algorithms::MEDIAN, 3);
	//std::cout << "\t Kernel size = 5 ..." << std::endl;
	//std::vector<double> median_5 = GetSigmaWithFilter(files, Denoising_Algorithms::MEDIAN, 5);
	//std::cout << "\t Kernel size = 7 ..." << std::endl;
	//std::vector<double> median_7 = GetSigmaWithFilter(files, Denoising_Algorithms::MEDIAN, 7);

	std::cout << "Calculating sigma vector for images modified with Gaussian Blurring algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<double> gaussian_3 = GetSigmaWithFilter(files, Denoising_Algorithms::GAUSSIAN, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<double> gaussian_5 = GetSigmaWithFilter(files, Denoising_Algorithms::GAUSSIAN, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<double> gaussian_7 = GetSigmaWithFilter(files, Denoising_Algorithms::GAUSSIAN, 7);

	std::cout << "Calculating sigma vector for images modified with Bilateral Filtering algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<double> bilateral_3 = GetSigmaWithFilter(files, Denoising_Algorithms::BILATERAL, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<double> bilateral_5 = GetSigmaWithFilter(files, Denoising_Algorithms::BILATERAL, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<double> bilateral_7 = GetSigmaWithFilter(files, Denoising_Algorithms::BILATERAL, 7);

	std::cout << "Started writing the file with noise calculation results ... " << std::endl;
	std::ofstream csv_file;
	csv_file.open("noise_results.csv");
	csv_file << "Filename, Initial Sigma, Sigma after Average3, Sigma after Average5, Sigma after Average7, " <<
							"Sigma after AdaptiveMedian, Sigma after Gaussian3, Sigma after Gaussian5, Sigma after Gaussian7, " << 
							"Sigma after Bilateral3, Sigma after Bilateral5, Sigma after Bilateral7, \n";

	std::cout << "\tWriting the results to the file..." << std::endl;
	for (int i = 0; i < initial.size(); i++)
		csv_file << files.at(i) << ", " << initial.at(i) << ", " << average_3.at(i) << ", " << average_5.at(i) << ", " << average_7.at(i) <<
								", " << adaptive_median.at(i) << ", " << gaussian_3.at(i) << ", " << gaussian_5.at(i) << ", " << gaussian_7.at(i) <<
								", " << bilateral_3.at(i) << ", " << bilateral_5.at(i) << ", " << bilateral_7.at(i) << ", \n";
	
	std::cout << "File was written successfully" << std::endl << std::endl;
	csv_file.close();
}

ALGORITHMSLIBRARY_API void WriteTimesCSVFile()
{
	using Duration = std::chrono::milliseconds;
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = GetFilePaths(RELATIVE_PATH);

	std::cout << "Calculating duration vector for images modified with Average Blurring algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<Duration> average_3 = GetAllRunningTimes(files, Denoising_Algorithms::AVERAGE, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<Duration> average_5 = GetAllRunningTimes(files, Denoising_Algorithms::AVERAGE, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<Duration> average_7 = GetAllRunningTimes(files, Denoising_Algorithms::AVERAGE, 7);

	std::cout << "Calculating sigma vector for images modified with Median Blurring algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<Duration> adaptive_median = GetAllRunningTimes(files, Denoising_Algorithms::MEDIAN, 3);

	std::cout << "Calculating sigma vector for images modified with Gaussian Blurring algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<Duration> gaussian_3 = GetAllRunningTimes(files, Denoising_Algorithms::GAUSSIAN, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<Duration> gaussian_5 = GetAllRunningTimes(files, Denoising_Algorithms::GAUSSIAN, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<Duration> gaussian_7 = GetAllRunningTimes(files, Denoising_Algorithms::GAUSSIAN, 7);

	std::cout << "Calculating sigma vector for images modified with Bilateral Filtering algorithm..." << std::endl;
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<Duration> bilateral_3 = GetAllRunningTimes(files, Denoising_Algorithms::BILATERAL, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<Duration> bilateral_5 = GetAllRunningTimes(files, Denoising_Algorithms::BILATERAL, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<Duration> bilateral_7 = GetAllRunningTimes(files, Denoising_Algorithms::BILATERAL, 7);

	std::cout << "Started writing the file with algorithm duration results ... " << std::endl;
	std::ofstream csv_file;
	csv_file.open("time_results.csv");
	csv_file << "Filename, Average3 Duration, Average5 Duration, Average7 Duration, " <<
		"Adaptive Median duration, Gaussian3 Duration, Gaussian5 Duration, Gaussian7 duration, " <<
		"Bilateral3 Duration, Bilateral5 Duration, Bilateral7 Duration, \n";

	std::cout << "\tWriting the results to the file..." << std::endl;
	for (int i = 0; i < files.size(); i++)
		csv_file << files.at(i) << ", " << average_3.at(i) << ", " << average_5.at(i) << ", " << average_7.at(i) <<
		", " << adaptive_median.at(i) << ", " << gaussian_3.at(i) << ", " << gaussian_5.at(i) << ", " << gaussian_7.at(i) <<
		", " << bilateral_3.at(i) << ", " << bilateral_5.at(i) << ", " << bilateral_7.at(i) << ", \n";

	std::cout << "File was written successfully" << std::endl << std::endl;
	csv_file.close();
}

ALGORITHMSLIBRARY_API int extractThresholdFromHistogram(cv::Mat& img, cv::Mat& histImage, uchar thresh)
{
	int bins = 256;
	std::vector<int> histogram(256, 0);
	for (int row = 0; row < img.rows; ++row)
	{
		uchar* imgRow = img.ptr<uchar>(row);
		for (int col = 0; col < img.cols; ++col)
		{
			if (imgRow[col] >= thresh)
				++histogram[(int)imgRow[col]];
		}
	}

	std::vector<int> cumulativeHistogram = histogram;
	for (int i = 1; i < histogram.size(); i++)
	{
		cumulativeHistogram[i] += cumulativeHistogram[i - 1];
	}

	cv::Point startPoint;
	startPoint.x = 0;
	startPoint.y = cumulativeHistogram[0];
	/*for (int i = 0; i < cumulativeHistogram.size(); i++)
	{
		if (cumulativeHistogram[i] > 0)
		{
			startPoint.x = i;
			startPoint.y = cumulativeHistogram[i];
			break;
		}
	}
	*/
	std::cout << "Start x: " << startPoint.x << std::endl;
	std::cout << "Start y: " << startPoint.y << std::endl;

	cv::Point endPoint;
	for (int i = cumulativeHistogram.size() - 1; i > 0; --i)
	{
		if (abs(cumulativeHistogram[i - 1] - cumulativeHistogram[i]) != 0)
		{
			endPoint.x = i;
			endPoint.y = cumulativeHistogram[i];
			break;
		}
	}
	std::cout << "End x: " << endPoint.x << std::endl;
	std::cout << "End y: " << endPoint.y << std::endl;

	// Ecuatia dreptei: y = m * x + n
	double m = 0;
	m = (endPoint.y - startPoint.y) / (endPoint.x - startPoint.x);

	double n = 0;
	n = ((startPoint.y * (startPoint.x + endPoint.x)) - (startPoint.x * (startPoint.y + endPoint.y))) / (endPoint.x - startPoint.x);

	std::vector<double> distancePointToLine(bins);
	for (int i = startPoint.x; i <= endPoint.x; i++)
	{
		cv::Point count = cv::Point(i, cumulativeHistogram[i]);
		distancePointToLine[i] = std::abs(count.y - (m * count.x) - n) / sqrt(1 + (m * m));
	}

	double maxDistance = distancePointToLine[0];
	int threshold = 0;

	for (int i = 1; i < distancePointToLine.size(); ++i)
	{
		if (maxDistance < distancePointToLine[i])
		{
			maxDistance = distancePointToLine[i];
			threshold = i;
		}
	}

	histImage = histogramDisplay(cumulativeHistogram, startPoint, endPoint, threshold);
	return threshold;
}

ALGORITHMSLIBRARY_API cv::Mat histogramDisplay(const std::vector<int>& histogram, const cv::Point& startPoint, const cv::Point& endPoint, int thresh)
{
	// draw the histograms
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histogram.size());

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	// find the maximum intensity element from histogram
	int max = *(histogram.end() - 1);
	//int min = *(histogram.begin());

	float yScaleFactor = hist_h / (max * 1.f);// -min * 1.f);

	// draw the intensity line for histogram
	for (int i = 0; i < histogram.size(); i++)
	{
		cv::Point startPt;
		startPt.x = bin_w * i;
		startPt.y = histogram[i] * yScaleFactor + hist_h - 1;

		cv::Point endPt;
		endPt.x = startPt.x;
		endPt.y = hist_h - histogram[i] * yScaleFactor;

		cv::line(histImage, startPt, endPt, cv::Scalar(255, 255, 255), 1, 8, 0);
	}

	cv::Point startPt;
	startPt.x = bin_w * startPoint.x;
	startPt.y = startPoint.y * yScaleFactor + hist_h - 1;

	cv::Point endPt;
	endPt.x = bin_w * endPoint.x;
	endPt.y = hist_h - endPoint.y * yScaleFactor;

	cv::line(histImage, startPt, endPt, cv::Scalar(0, 0, 255), 2, 8, 0);

	cv::line(histImage, cv::Point(bin_w * thresh, 0), cv::Point(bin_w * thresh, hist_h - 1), cv::Scalar(0, 255, 0), 2, 8, 0);

	return histImage;
}

ALGORITHMSLIBRARY_API cv::Mat SkullStripping_DynamicThreshold(cv::Mat& image)
{
	cv::Mat openedImage, cpyImage;
	image.copyTo(cpyImage);
	image.copyTo(openedImage);

	cv::Mat histImage;
	int threshold = extractThresholdFromHistogram(cpyImage, histImage);
	std::cout << "Threshold: " << threshold << std::endl;

	cv::threshold(cpyImage, openedImage, threshold, 255, cv::THRESH_BINARY);

	cv::Mat skullImage = cpyImage - openedImage;
	//cv::erode(skullImage, skullImage, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
	//cv::dilate(skullImage, skullImage, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

	return skullImage;
}

ALGORITHMSLIBRARY_API cv::Mat AdaptiveWindow_Threshold(cv::Mat& input)
{
	// Make all the pixels in the background 0 
	cv::Mat initial = RemoveBackground(input);

	// Declaring output cv::Mat
	cv::Mat output = cv::Mat(initial.rows, initial.cols, CV_8UC1);

	// Compute integral and squared integral image of the inital image
	cv::Mat sumImage, squaredSumImage;
	cv::integral(initial, sumImage, squaredSumImage, CV_64F);

	int maxStep = std::max(initial.rows, initial.cols);
	
	// Calculate the stdDev of the initial image
	const double* initialLastRowPtr = sumImage.ptr<double>(sumImage.rows - 1);
	double initialSum = initialLastRowPtr[sumImage.cols - 1];

	const double* initialSquaredSumLastRowPtr = squaredSumImage.ptr<double>(squaredSumImage.rows - 1);
	double initialSquaredSum = initialSquaredSumLastRowPtr[squaredSumImage.cols - 1];

	const long totalPixels = initial.rows * initial.cols;
	double initialMean = initialSum / totalPixels;
	double initialWindowVariance = (initialSquaredSum / totalPixels) - (initialMean) * (initialMean);
	double initialImageStdDev = sqrt(initialWindowVariance);

	const double weight = 0.1; // 10 %
	const double stdDevThreshold = weight * initialImageStdDev;

	for (int row = 0; row < input.rows; ++row)
	{
		const uchar* inputPtr = initial.ptr<uchar>(row);
		uchar* outputPtr = output.ptr<uchar>(row);

		for (int col = 0; col < input.cols; ++col)
		{
			// row = 80, col = 275
			int window_size = 3;
			int step = window_size / 2;
			double previousStdDevLog = 0;
			double currentStdDevLog = 0;
			double currentKernelStdDev = 0;
			double mean = 0;

			do
			{
				previousStdDevLog = currentStdDevLog;
				int rowUp = row - step;
				int rowDown = row + step;
				int colLeft = col - step;
				int colRight = col + step;

				// Checking borders
				if (rowUp < 0)
					rowUp = 0;
				if (rowDown >= input.rows)
					rowDown = input.rows - 1;
				if (colLeft < 0)
					colLeft = 0;
				if (colRight >= input.cols)
					colRight = input.cols - 1;

				const double* sumRowUpPtr = sumImage.ptr<double>(rowUp);
				const double* sumRowDownPtr = sumImage.ptr<double>(rowDown + 1);
				double sum = sumRowUpPtr[colLeft] + sumRowDownPtr[colRight + 1] - sumRowUpPtr[colRight + 1] - sumRowDownPtr[colLeft];

				const double* squaredSumRowUpPtr = squaredSumImage.ptr<double>(rowUp);
				const double* squaredSumRowDownPtr = squaredSumImage.ptr<double>(rowDown + 1);
				double squaredSum = squaredSumRowUpPtr[colLeft] + squaredSumRowDownPtr[colRight + 1] - squaredSumRowUpPtr[colRight + 1] - squaredSumRowDownPtr[colLeft];

				int windowWidth = colRight - colLeft + 1;
				int windowHeight = rowDown - rowUp + 1;
				int pixelsInWindow = windowHeight * windowWidth;
				mean = sum / pixelsInWindow;
				
				double currentWindowVariance = (squaredSum / pixelsInWindow) - (mean) * (mean);
				currentKernelStdDev = sqrt(currentWindowVariance);
				currentStdDevLog = currentKernelStdDev * log(windowWidth);
				
				++step;
			} while ((step <= maxStep) &&
							(currentStdDevLog > previousStdDevLog));
							//(currentKernelStdDev > stdDevThreshold));
							//(currentStdDevLog - previousStdDevLog >= epsilon));
			
			double threshold = mean;// +0.1 * currentKernelStdDev;
			
			if (inputPtr[col] >= threshold)
				outputPtr[col] = 0;
			else
				outputPtr[col] = 255;
		}
	}
	return output;
}

ALGORITHMSLIBRARY_API cv::Mat SkullStripping_AdaptiveWindow(cv::Mat& image)
{
	cv::Mat openedImage, cpyImage;
	image.copyTo(cpyImage);
	image.copyTo(openedImage);

	cv::Mat result = AdaptiveWindow_Threshold(image);
	return result;
}

ALGORITHMSLIBRARY_API cv::Mat SkullStripping_UsingMask(cv::Mat& image)
{
	cv::Mat hist;

	cv::Mat erodedImg;
	cv::erode(image, erodedImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	
	cv::Mat res = GradientTest(image);
	int thresh = extractThresholdFromHistogram(res, hist);

	cv::Mat thresholdRes = cv::Mat(res.rows, res.cols, CV_8UC1);

	for (int row = 0; row < res.rows; ++row)
	{
		uchar* resRow = res.ptr<uchar>(row);
		uchar* thResRow = thresholdRes.ptr<uchar>(row);

		for (int col = 0; col < res.cols; ++col)
		{
			if (resRow[col] >= thresh)
				thResRow[col] = 255;
			else
				thResRow[col] = 0;
		}
	}

	cv::Scalar mean, stdDev;
	cv::meanStdDev(image, mean, stdDev, thresholdRes);

	cv::Mat binaryFullMask = cv::Mat(image.rows, image.cols, CV_8UC1);
	for (int row = 0; row < res.rows; ++row)
	{
		uchar* imageRow = image.ptr<uchar>(row);
		uchar* binaryImgRow = binaryFullMask.ptr<uchar>(row);

		for (int col = 0; col < res.cols; ++col)
		{
			if (imageRow[col] >= (mean[0] - 1.5 * stdDev[0]))
				binaryImgRow[col] = 255;
			else
				binaryImgRow[col] = 0;
		}
	}

	cv::Mat binaryOuterMask = cv::Mat(image.rows, image.cols, CV_8UC1);
	for (int row = 0; row < res.rows; ++row)
	{
		uchar* imageRow = image.ptr<uchar>(row);
		uchar* binaryImgRow = binaryOuterMask.ptr<uchar>(row);

		for (int col = 0; col < res.cols; ++col)
		{
			if (imageRow[col] >= (mean[0]))
				binaryImgRow[col] = 255;
			else
				binaryImgRow[col] = 0;
		}
	}

	cv::Mat initialDifference = image - erodedImg;
	cv::Mat maskDifference = binaryFullMask - binaryOuterMask;


	cv::Mat imgAfterMask;
	cv::bitwise_and(image, maskDifference, imgAfterMask);
	return imgAfterMask;
}

ALGORITHMSLIBRARY_API cv::Mat SkullStripping_KMeans(cv::Mat& image)
{
	cv::Mat erodedImg;
	image.copyTo(erodedImg);
	cv::dilate(erodedImg, erodedImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13)));
	cv::erode(erodedImg, erodedImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));

	cv::Mat result = GradientTest(erodedImg);

	cv::Mat maskSkullStripping = SkullStripping_UsingMask(erodedImg);

	// K-means clustering for skull stripping
	cv::Mat samples = erodedImg.reshape(1, erodedImg.rows * erodedImg.cols);
	samples.convertTo(samples, CV_32FC1);

	cv::Mat bestLabels, centers;

	cv::kmeans(samples, 3, bestLabels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat labelsImg = bestLabels.reshape(1, erodedImg.rows);
	labelsImg.convertTo(labelsImg, CV_8U);

	float maxIntensityCenter = -5000;
	int threshValue = 0;
	for (int col = 0; col < centers.cols; ++col)
	{
		float* imgCenterCol = centers.ptr<float>(0);
		for (int row = 0; row < centers.rows; ++row)
		{
			maxIntensityCenter = std::max(maxIntensityCenter, imgCenterCol[row]);
			if (imgCenterCol[row] == maxIntensityCenter)
				threshValue = row;
		}
	}

	cv::Mat skullImage = cv::Mat::zeros(labelsImg.rows, labelsImg.cols, labelsImg.type());
	for (int row = 0; row < labelsImg.rows; ++row)
	{
		uchar* currentRow = labelsImg.ptr<uchar>(row);
		uchar* skullImageRow = skullImage.ptr<uchar>(row);

		for (int col = 0; col < labelsImg.cols; ++col)
		{
			if (currentRow[col] == threshValue)
				skullImageRow[col] = 255;
			else
				skullImageRow[col] = 0;
		}
	}

	cv::Mat copy;
	erodedImg.copyTo(copy);

	for (int row = 0; row < copy.rows; ++row)
	{
		uchar* currentRow = skullImage.ptr<uchar>(row);
		uchar* copyRow = copy.ptr<uchar>(row);

		for (int col = 0; col < copy.cols; ++col)
		{
			if (currentRow[col] == 255)
				copyRow[col] = 0;
		}
	}

	// Connected components on skull stripped image and finding the max area component
	std::pair<cv::Mat, int> connectedComp_result = ConnectedComponents(copy);
	int max_index = connectedComp_result.second;
	cv::Mat labeledImg = connectedComp_result.first;

	// Finding the contour of the brain region
	cv::Mat creier = ExtractTumorFromImage(labeledImg, max_index);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(creier, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Finding the center of mass of the max area component
	cv::Point center;
	int sumofx = 0, sumofy = 0;
	for (int i = 0; i < contours[0].size(); ++i) {
		sumofx = sumofx + contours[0][i].x;
		sumofy = sumofy + contours[0][i].y;
	}
	center.x = sumofx / contours[0].size();
	center.y = sumofy / contours[0].size();

	double maxDist = 0.0;
	double minDist = 5000.0;
	for (int i = 0; i < contours[0].size(); ++i)
	{
		double distCentrePoint = cv::norm(center - contours[0][i]);
		maxDist = std::max(maxDist, distCentrePoint);
		minDist = std::min(minDist, distCentrePoint);
	}

	// Aproximate a contour for the entire brain region
	std::vector<std::vector<cv::Point>> full_contour(contours.size());

	cv::convexHull(contours[0], full_contour[0]);

	cv::Mat drawn_contour = cv::Mat::zeros(image.size(), CV_8UC1);

	cv::drawContours(drawn_contour, full_contour, -1, 255, 1);// , cv::noArray(), 1);
	cv::fillConvexPoly(drawn_contour, full_contour[0], 255);

	// Extracting the brain region from the original image
	cv::Mat brain_image = cv::Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
	for (int row = 0; row < image.rows; ++row)
	{
		uchar* imgAfterMaskRow = image.ptr<uchar>(row);
		uchar* drawnContourRow = drawn_contour.ptr<uchar>(row);
		uchar* brain_imageRow = brain_image.ptr<uchar>(row);

		for (int col = 0; col < image.cols; ++col)
		{
			if (drawnContourRow[col] == 255)
				brain_imageRow[col] = imgAfterMaskRow[col];
		}
	}

	return brain_image;
}

ALGORITHMSLIBRARY_API cv::Mat GradientTest(cv::Mat& image)
{
	cv::Mat result;
	cv::morphologyEx(image, result, cv::MORPH_GRADIENT, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	return result;
}

ALGORITHMSLIBRARY_API cv::Mat ImageAfterOpening_UsingBinaryMask(cv::Mat& image)
{
	cv::Mat openedImage;
	image.copyTo(openedImage);

	cv::erode(openedImage, openedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
	cv::dilate(openedImage, openedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

	return openedImage;
}

ALGORITHMSLIBRARY_API cv::Mat KMeansClustering_Brain(cv::Mat& image)
{
	cv::Mat samples = image.reshape(1, image.rows * image.cols);
	samples.convertTo(samples, CV_32FC1);

	cv::Mat bestLabels, centers;

	cv::kmeans(samples, 3, bestLabels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

	cv::Mat labelsImg = bestLabels.reshape(1, image.rows);
	labelsImg.convertTo(labelsImg, CV_8U);

	float maxIntensityCenter = -5000;
	int threshValue = 0;
	for (int col = 0; col < centers.cols; ++col)
	{
		float* imgCenterCol = centers.ptr<float>(0);
		for (int row = 0; row < centers.rows; ++row)
		{
			maxIntensityCenter = std::max(maxIntensityCenter, imgCenterCol[row]);
			if (imgCenterCol[row] == maxIntensityCenter)
				threshValue = row;
		}
	}

	std::vector<int> areas(centers.rows, 0);
	for (int center_row = 0; center_row < centers.rows; ++center_row)
	{
		for (int row = 0; row < labelsImg.rows; ++row)
		{
			uchar* currentRow = labelsImg.ptr<uchar>(row);

			for (int col = 0; col < labelsImg.cols; ++col)
			{
				if (currentRow[col] == center_row)
					areas[center_row]++;
			}
		}
	}

	// Min area cluster is the tumor region
	int minElementIndex = std::min_element(areas.begin(), areas.end()) - areas.begin();

	// Extracting the min area cluster from the image
	cv::Mat tumorImg = cv::Mat::zeros(labelsImg.rows, labelsImg.cols, labelsImg.type());
	for (int row = 0; row < labelsImg.rows; ++row)
	{
		uchar* currentRow = labelsImg.ptr<uchar>(row);
		uchar* tumorImageRow = tumorImg.ptr<uchar>(row);

		for (int col = 0; col < labelsImg.cols; ++col)
		{
			if (currentRow[col] == minElementIndex)
				tumorImageRow[col] = 255;
			else
				tumorImageRow[col] = 0;
		}
	}

	return tumorImg;
}

ALGORITHMSLIBRARY_API std::pair<cv::Mat, int> ConnectedComponents(cv::Mat& image)
{
	cv::Mat labeledImage = cv::Mat(image.rows, image.cols, CV_8UC1);
	cv::Mat stats, centroids;
	int nrLabels = cv::connectedComponentsWithStats(image, labeledImage, stats, centroids);

	int max = -1;
	int indexMaxLabel = 0;
	std::vector<int> areas;
	for (int i = 1; i < nrLabels; ++i)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		areas.push_back(area);
		if (area > max)
		{
			max = area;
			indexMaxLabel = i;
		}
	}

	return std::pair(labeledImage, indexMaxLabel);
}

ALGORITHMSLIBRARY_API cv::Mat ExtractTumorArea(cv::Mat& image)
{
	std::pair<cv::Mat, int> pair_result = ConnectedComponents(image);
	cv::Mat labels_result = pair_result.first;
	int max_index_result = pair_result.second;

	cv::Mat tumora = ExtractTumorFromImage(labels_result, max_index_result);

	return tumora;
}

ALGORITHMSLIBRARY_API cv::Mat ExtractTumorFromImage(cv::Mat& image, const int indexMaxLabel)
{
	cv::Mat tumora = cv::Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
	for (int row = 0; row < image.rows; ++row)
	{
		int* imgRow = image.ptr<int>(row);
		uchar* tumoraRow = tumora.ptr<uchar>(row);

		for (int col = 0; col < image.cols; ++col)
		{
			if (imgRow[col] == indexMaxLabel)
				tumoraRow[col] = 255;
		}
	}

	return tumora;
}
