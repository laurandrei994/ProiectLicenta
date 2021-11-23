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
		cv::bilateralFilter(img, result, kernel_size, 75, 75);
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
	std::vector<double> results;
	for (std::string path : files)
	{
		cv::Mat initial = cv::imread(path);
		cv::Mat modified = ApplyDenoisingAlgorithm(initial, kernel_size, type);
		double sigma = EstimateNoise(modified);
		results.push_back(sigma);
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
	std::cout << "\t Kernel size = 3 ..." << std::endl;
	std::vector<double> median_3 = GetSigmaWithFilter(files, Denoising_Algorithms::MEDIAN, 3);
	std::cout << "\t Kernel size = 5 ..." << std::endl;
	std::vector<double> median_5 = GetSigmaWithFilter(files, Denoising_Algorithms::MEDIAN, 5);
	std::cout << "\t Kernel size = 7 ..." << std::endl;
	std::vector<double> median_7 = GetSigmaWithFilter(files, Denoising_Algorithms::MEDIAN, 7);

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
							"Sigma after Median3, Sigma after Median5, Sigma after Median7, " << 
							"Sigma after Gaussian3, Sigma after Gaussian5, Sigma after Gaussian7, " << 
							"Sigma after Bilateral3, Sigma after Bilateral5, Sigma after Bilateral7, \n";

	std::cout << "\tWriting the results to the file..." << std::endl;
	for (int i = 0; i < initial.size(); i++)
		csv_file << files.at(i) << ", " << initial.at(i) << ", " << average_3.at(i) << ", " << average_5.at(i) << ", " << average_7.at(i) <<
								", " << median_3.at(i) << ", " << median_5.at(i) << ", " << median_7.at(i) <<
								", " << gaussian_3.at(i) << ", " << gaussian_5.at(i) << ", " << gaussian_7.at(i) <<
								", " << bilateral_3.at(i) << ", " << bilateral_5.at(i) << ", " << bilateral_7.at(i) << ", \n";
	
	std::cout << "File was written successfully" << std::endl << std::endl;
	csv_file.close();
}
