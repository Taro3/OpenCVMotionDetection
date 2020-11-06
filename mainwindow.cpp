#include <QDebug>

#include <opencv2/opencv.hpp>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    cv::VideoCapture vc(0);
    if (!vc.isOpened())
        return;

    // create subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2> bs = cv::createBackgroundSubtractorMOG2(
                500         // number of previous frames used for background image
                , 16        // threshold for difference
                , true);    // whether to detect shadows

    qDebug() << "start";
    while (true) {
        if (cv::waitKey(1) >= 0)
            break;
        cv::Mat frame;
        cv::Mat mask;
        vc >> frame;
        if (frame.empty())
            break;
        bs->apply(frame, mask); // foreground extraction
        if (!mask.empty()) {
            // remove noise
            cv::threshold(mask, mask, 25, 255, cv::THRESH_BINARY);  // binarization
            int noiseSize = 9;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(noiseSize, noiseSize)); // create rectangle kernel
            cv::erode(mask, mask, kernel);
            kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(noiseSize, noiseSize));
            cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 3);

            // extract contour
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(
                        mask                        // image to extract.
                        , contours                  // list of detected contours
                        , cv::RETR_TREE             // all contours
                        , cv::CHAIN_APPROX_SIMPLE); // compresses horizontal, vertical, and diagonal segments and leaves only their end points.
            if (contours.size() > 0) {
                // draw detected rectangle
                cv::Scalar color = cv::Scalar(0, 0, 255);   // red
                for (size_t i = 0; i < contours.size(); ++i) {
                    cv::Rect rect = cv::boundingRect(contours[i]);
                    cv::rectangle(frame, rect, color, 1);
                }
            }
        }
        cv::imshow("Video", frame);
    }
    bs.release();
    vc.release();
    cv::destroyAllWindows();
    qDebug() << "finish";
}
