#include "Utils.h"

cv::Mat Utils::AverageFilter(cv::Mat& initial, const int kernel_size)
{
	cv::Mat result;
	cv::Point anchorPoint = cv::Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);

	cv::filter2D(initial, result, ddepth, kernel, anchorPoint, delta, cv::BORDER_DEFAULT);
	return result;
}

uchar Utils::AdaptiveProcess(cv::Mat& initial, const int row, const int col, int kernel_size, const int maxSize)
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
	//auto zxy = initial.at<uchar>(row, col);
	uchar* zxy = initial.ptr<uchar>(row);
	if (med > min && med < max)
	{
		if (zxy[col] > min && zxy[col] < max)
		{
			return zxy[col];
			//return zxy;
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

cv::Mat Utils::AdaptiveMedianFilter(cv::Mat& initial)
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
			//result.at<uchar>(j, i) = AdaptiveProcess(result, j, i, minSize, maxSize);
			ptr[i] = AdaptiveProcess(result, j, i, minSize, maxSize);
		}
	}
	return result;
}

cv::Mat Utils::GaussianFilter(cv::Mat& initial, const int kernel_size, const double sigma)
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
			//double sum[3] = { 0 };
			std::vector<double> sum(3, 0);
			for (int k = -border; k <= border; k++)
			{
				if (channels == 1)
				{
					//sum[0] += matrix[border + k] * result.at<uchar>(i, j + k);
					sum[0] += matrix[border + k] * row_ptr[j + k];
				}
				else if (channels == 3)
				{
					//int pixel = j * channels;
				
					//cv::Vec3b rgb = result.at<cv::Vec3b>(i, j + k);
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
				//result.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
				row_ptr[j] = static_cast<uchar>(sum[0]);
			else if (channels == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				row_ptr_vec[j] = rgb;
				//result.at<cv::Vec3b>(i, j) = rgb;
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
					//sum[0] += matrix[border + k] * result.at<uchar>(i + k, j);
					sum[0] += matrix[border + k] * row_ptr[j];
				}
				else if (channels == 3)
				{
					//cv::Vec3b rgb = result.at<cv::Vec3b>(i + k, j);
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
				//result.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
				ptr[j] = static_cast<uchar>(sum[0]);
			else if (channels == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				ptr_vec[j] = rgb;
				//result.at<cv::Vec3b>(i, j) = rgb;
			}
		}
	}
	delete[] matrix;
	return result;
}

cv::Mat Utils::BilateralFilter(cv::Mat& initial, const int kernel_size, const double space_sigma, const double color_sigma)
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

cv::Mat Utils::ApplyDenoisingAlgorithm(cv::Mat& img, const int kernel_size, Denoising_Algorithms type)
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

std::vector<std::string> Utils::GetFilePaths(const std::string& path)
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

double Utils::GetMSE(const cv::Mat& initial, const cv::Mat& modified)
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

std::vector<double> Utils::GetAllMSE(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
{
	std::vector<double> allMSE;
	for (std::string path : files)
	{
		cv::Mat initial = cv::imread(path);
		cv::Mat modified = Utils::ApplyDenoisingAlgorithm(initial, kernel_size, type);
		double mse = Utils::GetMSE(initial, modified);
		allMSE.push_back(mse);
	}
	return allMSE;
}

double Utils::EstimateNoise(const cv::Mat& img)
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
	sigma = sigma * sqrt(0.5 * std::numbers::pi) / (6 * (width - 2) * (height - 1));

	return sigma[0];
}

std::vector<double> Utils::GetSigmaWithFilter(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
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

std::chrono::milliseconds Utils::GetRunningTime(cv::Mat& img, const int kernel_size, const Denoising_Algorithms& type)
{
	using Duration = std::chrono::milliseconds;

	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat modified = ApplyDenoisingAlgorithm(img, kernel_size, type);
	auto stop = std::chrono::high_resolution_clock::now();
	const Duration duration = std::chrono::duration_cast<Duration>(stop - start);

	return duration;
}

std::vector<std::chrono::milliseconds> Utils::GetAllRunningTimes(const std::vector<std::string>& files, const Denoising_Algorithms& type, const int kernel_size)
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

void Utils::WriteMSECSVFile()
{
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = Utils::GetFilePaths(Utils::RELATIVE_PATH);

	std::cout << "Getting the MSE vector using Average Blurring algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Average_3 = Utils::GetAllMSE(files, Denoising_Algorithms::AVERAGE, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Average_5 = Utils::GetAllMSE(files, Denoising_Algorithms::AVERAGE, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Average_7 = Utils::GetAllMSE(files, Denoising_Algorithms::AVERAGE, 7);

	std::cout << "Getting the MSE vector using Median Blurring algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Median_3 = Utils::GetAllMSE(files, Denoising_Algorithms::MEDIAN, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Median_5 = Utils::GetAllMSE(files, Denoising_Algorithms::MEDIAN, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Median_7 = Utils::GetAllMSE(files, Denoising_Algorithms::MEDIAN, 7);

	std::cout << "Getting the MSE vector using Gaussian Blurring algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Gaussian_3 = Utils::GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Gaussian_5 = Utils::GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Gaussian_7 = Utils::GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 7);

	std::cout << "Getting the MSE vector using Bilateral Filtering algorithm..." << std::endl;
	std::cout << "\tKernel size = 3..." << std::endl;
	std::vector<double> allMSE_Bilateral_3 = Utils::GetAllMSE(files, Denoising_Algorithms::BILATERAL, 3);
	std::cout << "\tKernel size = 5..." << std::endl;
	std::vector<double> allMSE_Bilateral_5 = Utils::GetAllMSE(files, Denoising_Algorithms::BILATERAL, 5);
	std::cout << "\tKernel size = 7..." << std::endl;
	std::vector<double> allMSE_Bilateral_7 = Utils::GetAllMSE(files, Denoising_Algorithms::BILATERAL, 7);

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

void Utils::WriteNoiseCSVFile()
{
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = Utils::GetFilePaths(Utils::RELATIVE_PATH);

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

void Utils::WriteTimesCSVFile()
{
	using Duration = std::chrono::milliseconds;
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = Utils::GetFilePaths(Utils::RELATIVE_PATH);

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

