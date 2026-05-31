#ifndef REGISTERDIALOG_H
#define REGISTERDIALOG_H

#include <QDialog>
#include <QTimer>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

namespace Ui {
class RegisterDialog;
}

class RegisterDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RegisterDialog(QWidget *parent = nullptr);
    ~RegisterDialog();

    std::string newUserName;
    cv::Mat newUserEmbedding;
    bool isSuccess = false;

private slots:
    void updateFrame();
    void on_saveUserButton_clicked();

private:
    Ui::RegisterDialog *ui;

    cv::VideoCapture cap;
    QTimer *timer;

    cv::Ptr<cv::FaceDetectorYN> faceDetector;
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer;

    cv::Mat currentFrame;
    cv::Mat currentFaceBox;
    bool isFaceFound = false;

    bool isCountingDown = false;
    std::chrono::steady_clock::time_point startCaptureTime;
};

#endif // REGISTERDIALOG_H