#include "../../BrainTumorSegmentation/include/view.h"
#include "./ui_brainTumor.h"

MainWindow::MainWindow(QWidget* parent)
	:QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);
}

MainWindow::~MainWindow()
{
	if (ui)
		delete ui;
}