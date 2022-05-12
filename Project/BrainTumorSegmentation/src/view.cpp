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
	connect(ui->actionRemove_background_from_image, SIGNAL(triggered()), this, SLOT(RemoveBackground()));
	connect(ui->actionRemove_the_skull_from_the_image, SIGNAL(triggered()), this, SLOT(SkullStripping()));
	connect(ui->actionOpen_Skull_Stripped, SIGNAL(triggered()), this, SLOT(OpeningImage_UsingMask()));
	connect(ui->actionKmeans_clustering, SIGNAL(triggered()), this, SLOT(KMeans_clustering()));
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
	
	/*
	// Remove background from image.
	cv::Mat hist;

	cv::Mat otsuImg(grayImg.rows, grayImg.cols, CV_8UC1);
	double thresh = 0;
	double maxVal = 255;

	cv::threshold(grayImg, otsuImg, thresh, maxVal, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Mat imgAfterMask;

	cv::dilate(otsuImg, otsuImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25)));
	cv::erode(otsuImg, otsuImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25)));

	cv::bitwise_and(grayImg, otsuImg, imgAfterMask);

	int thresh2 = extractThresholdFromHistogram(imgAfterMask, hist, 1);

	cv::Mat binaryMask = cv::Mat(grayImg.rows, grayImg.cols, CV_8UC1);
	for (int row = 0; row < grayImg.rows; ++row)
	{
		uchar* imageRow = grayImg.ptr<uchar>(row);
		uchar* binaryImgRow = binaryMask.ptr<uchar>(row);

		for (int col = 0; col < grayImg.cols; ++col)
		{
			if (imageRow[col] >= thresh2)
				binaryImgRow[col] = 255;
			else
				binaryImgRow[col] = 0;
		}
	}
 // End Remove background from image
	// ---------------------------------------------------------------------------------------------------
	// Skull stripping
	cv::Mat erodedImg;
	imgAfterMask.copyTo(erodedImg);
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

	cv::Mat drawn_contour = cv::Mat::zeros(grayImg.size(), CV_8UC1);

	cv::drawContours(drawn_contour, full_contour, -1, 255, 1);// , cv::noArray(), 1);
	cv::fillConvexPoly(drawn_contour, full_contour[0], 255);

	// Extracting the brain region from the original image
	cv::Mat brain_image = cv::Mat::zeros(cv::Size(grayImg.rows, grayImg.cols), CV_8UC1);
	for (int row = 0; row < grayImg.rows; ++row)
	{
		uchar* imgAfterMaskRow = imgAfterMask.ptr<uchar>(row);
		uchar* drawnContourRow = drawn_contour.ptr<uchar>(row);
		uchar* brain_imageRow = brain_image.ptr<uchar>(row);

		for (int col = 0; col < grayImg.cols; ++col)
		{
			if (drawnContourRow[col] == 255)
				brain_imageRow[col] = imgAfterMaskRow[col];
		}
	}

	// End skull stripping
// -----------------------------------------------------------------------------
	// K-means clustering on brain image
	cv::Mat samples1 = brain_image.reshape(1, brain_image.rows * brain_image.cols);
	samples1.convertTo(samples1, CV_32FC1);

	cv::Mat bestLabels1, centers1;

	cv::kmeans(samples1, 3, bestLabels1, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers1);

	cv::Mat labelsImg1 = bestLabels1.reshape(1, brain_image.rows);
	labelsImg1.convertTo(labelsImg1, CV_8U);

	float maxIntensityCenter1 = -5000;
	int threshValue1 = 0;
	for (int col = 0; col < centers1.cols; ++col)
	{
		float* imgCenterCol = centers1.ptr<float>(0);
		for (int row = 0; row < centers1.rows; ++row)
		{
			maxIntensityCenter1 = std::max(maxIntensityCenter1, imgCenterCol[row]);
			if (imgCenterCol[row] == maxIntensityCenter1)
				threshValue = row;
		}
	}

	std::vector<int> areas(centers1.rows, 0);
	for (int center_row = 0; center_row < centers1.rows; ++center_row)
	{
		for (int row = 0; row < labelsImg1.rows; ++row)
		{
			uchar* currentRow = labelsImg1.ptr<uchar>(row);

			for (int col = 0; col < labelsImg1.cols; ++col)
			{
				if (currentRow[col] == center_row)
					areas[center_row]++;
			}
		}
	}

	// Min area cluster is the tumor region
	int minElementIndex = std::min_element(areas.begin(), areas.end()) - areas.begin();

	// Extracting the min area cluster from the image
	cv::Mat tumorImg = cv::Mat::zeros(labelsImg1.rows, labelsImg1.cols, labelsImg1.type());
	for (int row = 0; row < labelsImg1.rows; ++row)
	{
		uchar* currentRow = labelsImg1.ptr<uchar>(row);
		uchar* tumorImageRow = tumorImg.ptr<uchar>(row);

		for (int col = 0; col < labelsImg1.cols; ++col)
		{
			if (currentRow[col] == minElementIndex)
				tumorImageRow[col] = 255;
			else
				tumorImageRow[col] = 0;
		}
	}

	// ----------------------------------------------------

	// Pixels from image in the range of tumor
	cv::Mat openedImage;
	tumorImg.copyTo(openedImage);

	cv::erode(openedImage, openedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
	cv::dilate(openedImage, openedImage, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

	// ------------------------------------------------------------
	// Extracting the tumor from the image
	std::pair<cv::Mat, int> pair_result = ConnectedComponents(openedImage);
	cv::Mat labels_result = pair_result.first;
	int max_index_result = pair_result.second;

	cv::Mat tumora = ExtractTumorFromImage(labels_result, max_index_result);
	*/
	
	QImage convertedImage = Utils::ConvertMatToQImage(grayImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the grayscale algorithm");

}

void MainWindow::ApplyGaussianFilter()
{
	ui->nestStep->setText("Remove background from the image");
	cv::Mat modifiedImg = GaussianFilter(image, 5, 0.8);
	modifiedImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(modifiedImg);

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after applying the Gaussian Filter with a kernel of 5x5");
}

void MainWindow::RemoveBackground()
{
	ui->nestStep->setText("Apply skull stripping algorithm");
	cv::Mat modifiedImg = RemoveBackgroundFromImage(image);
	modifiedImg.copyTo(image);

	QImage convertedImage = Utils::ConvertMatToQImage(modifiedImg);
	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after removing the background");

}

void MainWindow::SkullStripping()
{
	ui->nestStep->setText("Begin segmentation process");
	cv::Mat brainImage = SkullStripping_KMeans(this->image);
	brainImage.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(brainImage);
	
	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after skull stripping");
}

void MainWindow::OpeningImage_UsingMask()
{
	ui->nestStep->setText("Connected Components With Stats");
	cv::Mat openedImage = ImageAfterOpening_UsingBinaryMask(this->image);
	openedImage.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(openedImage);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Image after opening");
}

void MainWindow::KMeans_clustering()
{
	ui->nestStep->setText("Apply opening on the image to remove unwanted pixels");
	cv::Mat tumorImg = KMeansClustering_Brain(this->image);
	tumorImg.copyTo(image);

	QImage convertedImg = Utils::ConvertMatToQImage(tumorImg);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Image after Kmeans algorithm");
}

void MainWindow::ExtractTumor()
{
	ui->nestStep->setText("Final Image");
	cv::Mat tumora = ExtractTumorArea(this->image);
	tumora.copyTo(this->image);

	QImage convertedImg = Utils::ConvertMatToQImage(tumora);

	ui->segmImg->setPixmap(QPixmap::fromImage(convertedImg).scaled(ui->segmImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->segmImgText->setText("Extracted tumor area from the MRI");

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
		ui->nestStep->setText("Remove background from image");
		break;
	case 3:
		RemoveBackground();
		index++;
		ui->nestStep->setText("Apply skull stripping algorithm");
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
