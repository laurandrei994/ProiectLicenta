#pragma once
#include <qimage.h>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QListWidget>
#include <QListWidgetItem>
#include <iostream>
#include <fstream>
#include <stack>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
	Q_OBJECT;

public:
	MainWindow(QWidget* parent = 0);
	~MainWindow();

private:
	Ui::MainWindow* ui;
};