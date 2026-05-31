#include "registerdialog.h"
#include "ui_registerdialog.h"
#include <QMessageBox>

RegisterDialog::RegisterDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RegisterDialog)
{
    ui->setupUi(this);

    faceDetector = cv::FaceDetectorYN::create("face_detection_yunet_2022mar.onnx", "", cv::Size(400, 400));
    faceRecognizer = cv::FaceRecognizerSF::create("face_recognition_sface_2021dec.onnx", "");

    cap.open(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 400);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 400);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &RegisterDialog::updateFrame);
    timer->start(33);
}

RegisterDialog::~RegisterDialog()
{
    timer->stop();
    cap.release();
    delete ui;
}

void RegisterDialog::updateFrame()
{
    cap >> currentFrame;
    if (currentFrame.empty()) return;

    cv::Mat displayFrame = currentFrame.clone();
    cv::Mat faces;

    faceDetector->setInputSize(currentFrame.size());
    faceDetector->detect(currentFrame, faces);

    isFaceFound = false;

    if (!faces.empty()) {
        float conf = faces.at<float>(0, 14);
        if (conf > 0.7) {
            isFaceFound = true;
            currentFaceBox = faces.row(0).clone();

            int x = int(faces.at<float>(0, 0));
            int y = int(faces.at<float>(0, 1));
            int w = int(faces.at<float>(0, 2));
            int h = int(faces.at<float>(0, 3));
            cv::rectangle(displayFrame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
        }
    }

    bool hasName = !ui->nameInput->text().trimmed().isEmpty();

    // Auto-Capture
    if (isFaceFound && hasName) {
        if (!isCountingDown) {
            isCountingDown = true;
            startCaptureTime = std::chrono::steady_clock::now();
        } else {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startCaptureTime).count();
            int remaining = 3 - elapsed; // 3 secsz

            if (remaining > 0) {
                std::string text = "Scanning...";
                cv::putText(displayFrame, text, cv::Point(20, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
            } else {
                on_saveUserButton_clicked();
                return;
            }
        }
    } else {
        isCountingDown = false;

        if (!isFaceFound) {
            cv::putText(displayFrame, "Please look at the camera", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::cvtColor(displayFrame, displayFrame, cv::COLOR_BGR2RGB);
    QImage img((const unsigned char*)(displayFrame.data), displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888);
    ui->cameraLabel->setPixmap(QPixmap::fromImage(img.copy()));
}

void RegisterDialog::on_saveUserButton_clicked()
{
    if (!isFaceFound) return;

    cv::Mat aligned_face;
    faceRecognizer->alignCrop(currentFrame, currentFaceBox, aligned_face);
    faceRecognizer->feature(aligned_face, newUserEmbedding);

    newUserName = ui->nameInput->text().toStdString();
    isSuccess = true;

    QMessageBox::information(this, "Success!", "User saved!");
    this->accept();
}