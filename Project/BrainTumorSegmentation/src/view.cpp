#include "../../BrainTumorSegmentation/include/view.h"
#include "./ui_brainTumor.h"
#include "../include/Utils.h"
#include "../../Algorithms/include/Algorithms.h"

#include <qfile.h>
#include <qfiledialog.h>

MainWindow::MainWindow(QWidget* parent)
	:QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	CreateActions();
}

MainWindow::~MainWindow()
{
	if (ui)
		delete ui;
}

void MainWindow::CreateActions()
{
	connect(ui->actionOpen_Image, SIGNAL(triggered()), this, SLOT(OpenFile()));
	connect(ui->actionConvert_To_Grayscale, SIGNAL(triggered()), this, SLOT(ConvertToGrayScale()));
	connect(ui->actionClose, SIGNAL(triggered()), this, SLOT(close()));
}

//SLOTS
void MainWindow::OpenFile()
{
	QString filter = "All files (*.*);;JPEG(*.jpg);;PNG(*.png);;TIF(*.tif)";
	QFile file(QString::fromStdString(Utils::dataSetPath));
	QString filepath = QFileDialog::getOpenFileName(this, tr("Open File"), QString::fromStdString(Utils::dataSetPath), filter);

	if (filepath.isEmpty())
	{
		std::cout << "File is not an image!!" << std::endl;
		return;
	}
	image = Utils::ReadImage(filepath.toStdString());
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	QImage convertedImage = Utils::ConvertMatToQImage(image);
	qImage = convertedImage;
	
	//ui->inputImg->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

	ui->inputImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->inputImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->inputImgText->setText("Initial Image");
}

void MainWindow::ConvertToGrayScale()
{
	cv::Mat grayImg = GrayScale_Average(image);
	image = grayImg;

	QImage convertedImage = Utils::ConvertMatToQImage(grayImg);
	qImage = convertedImage;

	ui->preprocImg->setPixmap(QPixmap::fromImage(convertedImage).scaled(ui->preprocImg->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
	ui->preprocImgText->setText("Image after aplying the grayscale algorithm");
}

