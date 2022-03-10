#include "Algorithms.h"
#include "../../BrainTumorSegmentation/include/view.h"

#include <QApplication>

int main(int argc, char* argv[]) {

	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}