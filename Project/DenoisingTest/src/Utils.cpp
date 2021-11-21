#include "Utils.h"

cv::Mat Utils::ApplyDenoisingAlgorithm(const cv::Mat& img, const int kernel_size, Denoising_Algorithms type)
{
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

void Utils::WriteCSVFile()
{
	std::cout << "Getting the filepaths from the TestData folder..." << std::endl;
	std::vector<std::string> files = Utils::GetFilePaths(Utils::RELATIVE_PATH);

	std::cout << "Getting the MSE vector using Average Blurring algorithm with a kernel size of 3..." << std::endl;
	std::vector<double> allMSE_Average_3 = Utils::GetAllMSE(files, Denoising_Algorithms::AVERAGE, 3);
	std::cout << "Getting the MSE vector using Average Blurring algorithm with a kernel size of 5..." << std::endl;
	std::vector<double> allMSE_Average_5 = Utils::GetAllMSE(files, Denoising_Algorithms::AVERAGE, 5);
	std::cout << "Getting the MSE vector using Average Blurring algorithm with a kernel size of 7..." << std::endl;
	std::vector<double> allMSE_Average_7 = Utils::GetAllMSE(files, Denoising_Algorithms::AVERAGE, 7);

	std::cout << "Getting the MSE vector using Median Blurring algorithm with a kernel size of 3..." << std::endl;
	std::vector<double> allMSE_Median_3 = Utils::GetAllMSE(files, Denoising_Algorithms::MEDIAN, 3);
	std::cout << "Getting the MSE vector using Median Blurring algorithm with a kernel size of 5..." << std::endl;
	std::vector<double> allMSE_Median_5 = Utils::GetAllMSE(files, Denoising_Algorithms::MEDIAN, 5);
	std::cout << "Getting the MSE vector using Median Blurring algorithm with a kernel size of 7..." << std::endl;
	std::vector<double> allMSE_Median_7 = Utils::GetAllMSE(files, Denoising_Algorithms::MEDIAN, 7);

	std::cout << "Getting the MSE vector using Gaussian Blurring algorithm with a kernel size of 3..." << std::endl;
	std::vector<double> allMSE_Gaussian_3 = Utils::GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 3);
	std::cout << "Getting the MSE vector using Gaussian Blurring algorithm with a kernel size of 5..." << std::endl;
	std::vector<double> allMSE_Gaussian_5 = Utils::GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 5);
	std::cout << "Getting the MSE vector using Gaussian Blurring algorithm with a kernel size of 7..." << std::endl;
	std::vector<double> allMSE_Gaussian_7 = Utils::GetAllMSE(files, Denoising_Algorithms::GAUSSIAN, 7);

	std::cout << "Getting the MSE vector using Bilateral Filtering algorithm with a kernel size of 3..." << std::endl;
	std::vector<double> allMSE_Bilateral_3 = Utils::GetAllMSE(files, Denoising_Algorithms::BILATERAL, 3);
	std::cout << "Getting the MSE vector using Bilateral Filtering algorithm with a kernel size of 5..." << std::endl;
	std::vector<double> allMSE_Bilateral_5 = Utils::GetAllMSE(files, Denoising_Algorithms::BILATERAL, 5);
	std::cout << "Getting the MSE vector using Bilateral Filtering algorithm with a kernel size of 7..." << std::endl;
	std::vector<double> allMSE_Bilateral_7 = Utils::GetAllMSE(files, Denoising_Algorithms::BILATERAL, 7);

	std::cout << "Started writing the file.. " << std::endl;
	std::ofstream csv_file;
	csv_file.open("mse_results.csv");
	csv_file << "Filename, Algorithm, Kernel Size, MSE, \n";
	
	std::cout << "\tWriting the Average Blurring results.." << std::endl;
	for (int i = 0; i < allMSE_Average_3.size(); i++)
		csv_file << files.at(i) << ", " << "Average Blurring, " << "3x3," << allMSE_Average_3.at(i) << ", \n";
	for (int i = 0; i < allMSE_Average_5.size(); i++)
		csv_file << files.at(i) << ", " << "Average Blurring, " << "5x5," << allMSE_Average_5.at(i) << ", \n";
	for (int i = 0; i < allMSE_Average_7.size(); i++)
		csv_file << files.at(i) << ", " << "Average Blurring, " << "7x7," << allMSE_Average_7.at(i) << ", \n";
	
	std::cout << "\tWriting the Median Blurring results.." << std::endl;
	for (int i = 0; i < allMSE_Median_3.size(); i++)
		csv_file << files.at(i) << ", " << "Median Blurring, " << "3x3, " << allMSE_Median_3.at(i) << ", \n";
	for (int i = 0; i < allMSE_Median_5.size(); i++)
		csv_file << files.at(i) << ", " << "Median Blurring, " << "5x5, " << allMSE_Median_5.at(i) << ", \n";
	for (int i = 0; i < allMSE_Median_7.size(); i++)
		csv_file << files.at(i) << ", " << "Median Blurring, " << "7x7, " << allMSE_Median_7.at(i) << ", \n";
	
	std::cout << "\tWriting the Gaussian Blurring results.." << std::endl;
	for (int i = 0; i < allMSE_Gaussian_3.size(); i++)
		csv_file << files.at(i) << ", " << "Gaussian Blurring, " << "3x3, " << allMSE_Gaussian_3.at(i) << ", \n";
	for (int i = 0; i < allMSE_Gaussian_5.size(); i++)
		csv_file << files.at(i) << ", " << "Gaussian Blurring, " << "5x5, " << allMSE_Gaussian_5.at(i) << ", \n";
	for (int i = 0; i < allMSE_Gaussian_7.size(); i++)
		csv_file << files.at(i) << ", " << "Gaussian Blurring, " << "7x7, " << allMSE_Gaussian_7.at(i) << ", \n";
	
	std::cout << "\tWriting the Bilateral Filtering results.." << std::endl;
	for (int i = 0; i < allMSE_Bilateral_3.size(); i++)
		csv_file << files.at(i) << ", " << "Bialteral Filtering, " << "3x3, " << allMSE_Bilateral_3.at(i) << ", \n";
	for (int i = 0; i < allMSE_Bilateral_5.size(); i++)
		csv_file << files.at(i) << ", " << "Bialteral Filtering, " << "5x5, " << allMSE_Bilateral_5.at(i) << ", \n";
	for (int i = 0; i < allMSE_Bilateral_7.size(); i++)
		csv_file << files.at(i) << ", " << "Bialteral Filtering, " << "7x7, " << allMSE_Bilateral_7.at(i) << ", \n";

	std::cout << "File was written successfully" << std::endl;
	csv_file.close();
}
