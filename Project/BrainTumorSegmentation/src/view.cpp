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
	cv::Mat grayImg = GrayScale_Average(image);
	grayImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(grayImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the grayscale algorithm");
}

void MainWindow::ApplyGaussianFilter()
{
	cv::Mat modifiedImg = GaussianFilter(image, 5, 0.8);
	modifiedImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(modifiedImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the Gaussian Filter with a kernel of 5x5");
}

void MainWindow::SkullStripping()
{
	//cv::Mat skullImage = SkullStripping_DynamicThreshold(this->image);
	cv::Mat skullImage = SkullStripping_AdaptiveWindow(this->image);
	//cv::Mat skullImage2 = SkullStripping_AdaptiveWindow(skullImage);
	QImage convertedImg = Utils::ConvertMatToQImage(skullImage);

	//cv::erode(skullImage2, skullImage2,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(7,7)));
	//cv::dilate(skullImage2, skullImage2,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after skull stripping");
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
