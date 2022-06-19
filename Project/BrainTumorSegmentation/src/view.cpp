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
	DisableMenuItems();
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
	connect(ui->actionRemove_background_from_image, SIGNAL(triggered()), this, SLOT(RemoveBackground()));
	connect(ui->actionRemove_the_skull_from_the_image, SIGNAL(triggered()), this, SLOT(SkullStripping()));
	connect(ui->actionOpen_Skull_Stripped, SIGNAL(triggered()), this, SLOT(OpeningImage_UsingMask()));
	connect(ui->actionKmeans_clustering, SIGNAL(triggered()), this, SLOT(KMeans_clustering()));
	connect(ui->actionExtract_the_tumor_from_the_image, SIGNAL(triggered()), this, SLOT(ExtractTumor()));
	connect(ui->actionFinal_Image, SIGNAL(triggered()), this, SLOT(ConstructResult()));

	connect(ui->nestStep, SIGNAL(clicked()), this, SLOT(NextStepClick()));
}

//SLOTS
void MainWindow::OpenFile()
{
	ClearLabels();
	DisableMenuItems();
	ui->nestStep->setDisabled(false);
	ui->actionConvert_To_Grayscale->setDisabled(false);

	ui->nestStep->setText("Convert image to grayscale");
	index = 1;
	cv::Mat img = OpenImage();

	QImage convertedImage = Utils::ConvertMatToQImage(img);

	ui->inputImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->inputImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->inputImgText->setText("Initial Image");
}

void MainWindow::OpenRandomFile()
{
	ClearLabels();
	DisableMenuItems();
	ui->nestStep->setDisabled(false);
	ui->actionConvert_To_Grayscale->setDisabled(false);

	ui->nestStep->setText("Convert image to grayscale");
	index = 1;
	cv::Mat img = OpenRandomImage();

	QImage convertedImage = Utils::ConvertMatToQImage(img);

	ui->inputImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->inputImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->inputImgText->setText("Initial Image");
}

void MainWindow::ConvertToGrayScale()
{
	ui->nestStep->setText("Apply Gaussian Blurring algorithm");
	ui->actionGaussian_Filter->setDisabled(false);
	cv::Mat grayImg = GrayScale_Average(image);
	grayImg.copyTo(image);
	
	QImage convertedImage = Utils::ConvertMatToQImage(grayImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the grayscale algorithm");
}

void MainWindow::ApplyGaussianFilter()
{
	ui->nestStep->setText("Remove background from the image");
	ui->actionRemove_background_from_image->setDisabled(false);
	cv::Mat modifiedImg = GaussianFilter(image, 5, 0.8);
	modifiedImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(modifiedImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the Gaussian Filter with a kernel of 5x5");
}

void MainWindow::RemoveBackground()
{
	ui->nestStep->setText("Apply skull stripping algorithm");
	ui->actionRemove_the_skull_from_the_image->setDisabled(false);
	cv::Mat modifiedImg = RemoveBackgroundFromImage(image);
	modifiedImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(modifiedImg);
	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after removing the background");
}

void MainWindow::SkullStripping()
{
	ui->nestStep->setText("Begin segmentation process");
	ui->actionKmeans_clustering->setDisabled(false);
	cv::Mat brainImage = SkullStripping_KMeans(this->image);
	brainImage.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(brainImage);
	
	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->preprocImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after skull stripping");
}

void MainWindow::OpeningImage_UsingMask()
{
	ui->nestStep->setText("Connected Components With Stats");
	ui->actionExtract_the_tumor_from_the_image->setDisabled(false);
	cv::Mat openedImage = ImageAfterOpening_UsingBinaryMask(this->image);
	openedImage.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(openedImage);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Image after opening");
}

void MainWindow::KMeans_clustering()
{
	ui->nestStep->setText("Apply opening on the image to remove unwanted pixels");
	ui->actionOpen_Skull_Stripped->setDisabled(false);
	cv::Mat tumorImg = KMeansClustering_Brain(this->image);
	tumorImg.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(tumorImg);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Image after Kmeans algorithm");
}

void MainWindow::ExtractTumor()
{
	ui->nestStep->setText("Final Image");
	ui->actionFinal_Image->setDisabled(false);
	cv::Mat tumora = ExtractTumorArea(this->image);
	tumora.copyTo(this->image);
	
	QImage convertedImg = Utils::ConvertMatToQImage(tumora);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Extracted tumor area from the MRI");
}

void MainWindow::ConstructResult()
{
	ui->nestStep->setDisabled(true);
	DisableMenuItems();
	cv::Mat result = ConstructFinalImage(this->image, this->initialImage);
	result.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(result);

	ui->resultImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->resultImg->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	ui->resultImgText->setText("Image with the tumor highlighted");
}

void MainWindow::NextStepClick()
{
	switch (index)
	{
	case 0:
		OpenFile();
		if (index < 1)
			index++;
		ui->nestStep->setDisabled(false);
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
		ui->nestStep->setText("Remove background from image");
		break;
	case 3:
		RemoveBackground();
		index++;
		ui->nestStep->setText("Apply skull stripping algorithm");
		break;
	case 4:
		SkullStripping();
		index++;
		ui->nestStep->setText("Begin segmentation process");
		break;
	case 5:
		KMeans_clustering();
		index++;
		ui->nestStep->setText("Apply opening on the image to remove unwanted pixels");
		break;
	case 6:
		OpeningImage_UsingMask();
		index++;
		ui->nestStep->setText("Extract tumor area from image using Connected Components Algorithm");
		break;
	case 7:
		ExtractTumor();
		index++;
		ui->nestStep->setText("Final image");
		break;
	case 8:
		ConstructResult();
		index++;
		ui->nestStep->setDisabled(true);
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

	image.copyTo(initialImage);
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

	image.copyTo(initialImage);
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

void MainWindow::DisableMenuItems()
{
	ui->actionConvert_To_Grayscale->setDisabled(true);
	ui->actionGaussian_Filter->setDisabled(true);
	ui->actionRemove_background_from_image->setDisabled(true);
	ui->actionRemove_the_skull_from_the_image->setDisabled(true);
	ui->actionOpen_Skull_Stripped->setDisabled(true);
	ui->actionKmeans_clustering->setDisabled(true);
	ui->actionExtract_the_tumor_from_the_image->setDisabled(true);
	ui->actionFinal_Image->setDisabled(true);
}
