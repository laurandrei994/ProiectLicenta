#include "Algorithms.h"
#include "../../BrainTumorSegmentation/include/view.h"

#include <QApplication>

int main(int argc, char* argv[]) {
	/*const std::string FILEPATH = "E:\\FACULTATE\\UniTBv\\Anul III\\Licenta\\ProiectLicenta\\TestData\\Te-gl_0011.jpg";
	cv::Mat image = cv::imread(FILEPATH);

	cv::Mat image2 = GaussianFilter(image, 3, 0.5);
	cv::imshow("Image", image2);
	cv::waitKey(0);*/

	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}