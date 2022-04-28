#include "../../BrainTumorSegmentation/include/view.h"
#include "./ui_brainTumor.h"
#include "../include/Utils.h"
#include "../../Algorithms/include/Algorithms.h"

#include <qfile.h>
#include <qfiledialog.h>

#include <random>

MainWindow::MainWindow(QWidget* parent)
	:QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	index = 0;
	maxLabelIndex = 0;
	CreateActions();
	ClearLabels();
}

MainWindow::~MainWindow()
{
	if (ui)
		delete ui;
}

void MainWindow::CreateActions()
{
	connect(ui->actionOpen_Image, SIGNAL(triggered()), this, SLOT(OpenFile()));
	connect(ui->actionOpen_Random_Image, SIGNAL(triggered()), this, SLOT(OpenRandomFile()));
	connect(ui->actionConvert_To_Grayscale, SIGNAL(triggered()), this, SLOT(ConvertToGrayScale()));
	connect(ui->actionClose, SIGNAL(triggered()), this, SLOT(close()));
	connect(ui->actionGaussian_Filter, SIGNAL(triggered()), this, SLOT(ApplyGaussianFilter()));
	connect(ui->actionRemove_the_skull_from_the_image, SIGNAL(triggered()), this, SLOT(SkullStripping()));
	connect(ui->actionOpen_Skull_Stripped, SIGNAL(triggered()), this, SLOT(OpeningImage_UsingMask()));
	connect(ui->actionConnected_Components, SIGNAL(triggered()), this, SLOT(ConnectedComponentsWithStats()));
	connect(ui->actionExtract_the_tumor_from_the_image, SIGNAL(triggered()), this, SLOT(ExtractTumor()));

	connect(ui->nestStep, SIGNAL(clicked()), this, SLOT(NextStepClick()));
}

//SLOTS
void MainWindow::OpenFile()
{
	ClearLabels();
	ui->nestStep->setText("Convert image to grayscale");
	index = 1;
	cv::Mat img = OpenImage();
	QImage convertedImage = Utils::ConvertMatToQImage(img);

	ui->inputImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->inputImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->inputImgText->setText("Initial Image");
}

void MainWindow::OpenRandomFile()
{
	ClearLabels();
	ui->nestStep->setText("Convert image to grayscale");
	index = 1;
	cv::Mat img = OpenRandomImage();
	QImage convertedImage = Utils::ConvertMatToQImage(img);

	ui->inputImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->inputImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->inputImgText->setText("Initial Image");
}

void MainWindow::ConvertToGrayScale()
{
	ui->nestStep->setText("Apply Gaussian Blurring algorithm");
	cv::Mat grayImg = GrayScale_Average(image);
	grayImg.copyTo(image);
	
	//cv::Mat res = GradientTest(image);
	//cv::Mat hist;
	//cv::Mat imgAfterMask = SkullStripping_UsingMask(grayImg);
	/*
	cv::Mat erodedImg;
	cv::erode(grayImg, erodedImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	///
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

	cv::Mat initialDifference = grayImg - erodedImg;
	cv::Mat maskDifference = binaryFullMask - binaryOuterMask;


	cv::Mat imgAfterMask; 
	cv::bitwise_and(grayImg, maskDifference, imgAfterMask);  // skull stripping
	

	int thresh2 = extractThresholdFromHistogram(imgAfterMask, hist, 1);

	cv::Mat binaryImgAfterMask = cv::Mat(imgAfterMask.rows, imgAfterMask.cols, CV_8UC1);
	for (int row = 0; row < res.rows; ++row)
	{
		uchar* imgAfterMaskRow = imgAfterMask.ptr<uchar>(row);
		uchar* binaryImgAfterMaskRow = binaryImgAfterMask.ptr<uchar>(row);

		for (int col = 0; col < res.cols; ++col)
		{
			if (imgAfterMaskRow[col] == 0)
			{
				binaryImgAfterMaskRow[col] = 0;
				continue;
			}
			// 0/255 pentru gl
			// 255/0 pt me
			if (imgAfterMaskRow[col] >= thresh2)
				binaryImgAfterMaskRow[col] = 0;
			else
				binaryImgAfterMaskRow[col] = 255;
		}
	}

	cv::Mat copy;
	binaryImgAfterMask.copyTo(copy);

	cv::erode(copy, copy, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
	cv::dilate(copy, copy, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
	

	cv::Mat labeledImage = cv::Mat(imgAfterMask.rows, imgAfterMask.cols, CV_8UC1);
	cv::Mat stats, centroids;
	int nrLabels = cv::connectedComponentsWithStats(copy, labeledImage,stats,centroids);

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
	*/
	//cv::Mat copy = ImageAfterOpening_UsingBinaryMask(imgAfterMask);
	//cv::Mat labeledImage = cv::Mat(imgAfterMask.rows, imgAfterMask.cols, CV_8UC1);
	//std::pair<cv::Mat, int> pair = ConnectedComponents(copy);
	//cv::Mat labeledImage = pair.first;
	//int indexMaxLabel = pair.second;
	/* cv::Mat tumora = cv::Mat::zeros(cv::Size(labeledImage.rows, labeledImage.cols), CV_8UC1);
	for (int row = 0; row < labeledImage.rows; ++row)
	{
		int* imgRow = labeledImage.ptr<int>(row);
		uchar* tumoraRow = tumora.ptr<uchar>(row);

		for (int col = 0; col < labeledImage.cols; ++col)
		{
			if (imgRow[col] == indexMaxLabel)
				tumoraRow[col] = 255;
		}
	}
	//cv::Mat tumora = ExtractTumorFromImage(labeledImage, indexMaxLabel);
	*/

	QImage convertedImage = Utils::ConvertMatToQImage(grayImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the grayscale algorithm");
}

void MainWindow::ApplyGaussianFilter()
{
	ui->nestStep->setText("Apply skull stripping algorithm");
	cv::Mat modifiedImg = GaussianFilter(image, 5, 0.8);
	modifiedImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(modifiedImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the Gaussian Filter with a kernel of 5x5");
}

void MainWindow::SkullStripping()
{
	ui->nestStep->setText("Begin segmentation process");
	cv::Mat skullImage = SkullStripping_UsingMask(this->image);
	//cv::Mat skullImage = SkullStripping_DynamicThreshold(this->image);

	QImage convertedImg = Utils::ConvertMatToQImage(skullImage);
	skullImage.copyTo(image);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after skull stripping");
}

void MainWindow::OpeningImage_UsingMask()
{
	ui->nestStep->setText("Connected Components With Stats");
	cv::Mat openedImage = ImageAfterOpening_UsingBinaryMask(this->image);

	QImage convertedImg = Utils::ConvertMatToQImage(openedImage);
	openedImage.copyTo(this->image);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Image after opening using binary mask");
}

void MainWindow::ConnectedComponentsWithStats()
{
	ui->nestStep->setText("Extract the tumor from the image");
	std::pair<cv::Mat, int> labeledImg_MaxIndex = ConnectedComponents(this->image);

	this->labeledImg = labeledImg_MaxIndex.first;
	this->maxLabelIndex = labeledImg_MaxIndex.second;

	QImage convertedImg = Utils::ConvertMatToQImage(this->image);

	std::cout << "MAX LABEL INDEX: " << maxLabelIndex << std::endl;

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Labeled Image");
}

void MainWindow::ExtractTumor()
{
	ui->nestStep->setText("Final Image");
	cv::Mat tumora = ExtractTumorFromImage(this->labeledImg, this->maxLabelIndex);

	QImage convertedImg = Utils::ConvertMatToQImage(tumora);
	tumora.copyTo(this->image);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Extracted tumor from the MRI");

}

void MainWindow::NextStepClick()
{
	switch (index)
	{
	case 0:
		OpenFile();
		if (index < 1)
			index++;
		ui->nestStep->setText("Convert image to grayscale");
		break;
	case 1:
		ConvertToGrayScale();
		index++;
		ui->nestStep->setText("Apply Gaussian Blurring algorithm");
		break;
	case 2:
		ApplyGaussianFilter();
		index++;
		ui->nestStep->setText("Apply skull stripping algorithm");
		break;
	case 3:
		SkullStripping();
		index++;
		ui->nestStep->setText("Begin segmentation process");
		break;
	case 4: 
		OpeningImage_UsingMask();
		index++;
		ui->nestStep->setText("Connected Components With Stats");
		break;
	case 5:
		ConnectedComponentsWithStats();
		index++;
		ui->nestStep->setText("Extract the tumor from the image");
		break;
	case 6:
		ExtractTumor();
		index++;
		ui->nestStep->setText("Final image");
		break;
	default:
		break;
	}
}

cv::Mat MainWindow::OpenImage()
{
	QString filter = "All files (*.*);;JPEG(*.jpg);;PNG(*.png);;TIF(*.tif)";
	QFile file(QString::fromStdString(Utils::dataSetPath));
	QString filepath = QFileDialog::getOpenFileName(this, tr("Open File"), QString::fromStdString(Utils::dataSetPath), filter);

	if (filepath.isEmpty())
	{
		std::cout << "File is not an image!!" << std::endl;
		ui->inputImgText->setText("You have not selected an image!!");
		return cv::Mat();
	}
	image = Utils::ReadImage(filepath.toStdString());
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	
	return image;
}

cv::Mat MainWindow::OpenRandomImage()
{
	QDir source(QString::fromStdString(Utils::dataSetPath));
	if (!source.exists())
	{
		std::cout << "Bad path!!" << std::endl;
		ui->inputImgText->setText("Can't open random image!!");
		return cv::Mat();
	}
	//Generating a list with the name of all the files in the source folder
	QStringList fileList = source.entryList();

	//Generating a random number in range (0, lenght of list) for the random image
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(0, fileList.length());
	int generatedNumber = distr(gen);

	QString filepath = QString::fromStdString(Utils::dataSetPath) + fileList[generatedNumber];
	std::cout << filepath.toStdString() << std::endl;

	image = Utils::ReadImage(filepath.toStdString());
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	return image;
}

void MainWindow::ClearLabels()
{
	ui->preprocImg->clear();
	ui->preprocImgText->clear();
	ui->segmImg->clear();
	ui->segmImgText->clear();
	ui->resultImg->clear();
	ui->resultImgText->clear();
	ui->nestStep->setText("Open Image");
}
