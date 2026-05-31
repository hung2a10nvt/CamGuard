#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QPixmap>

// --- Core C++ & OpenCV ---
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <string>

// --- Custom Modules ---
#include "BoundedQueue.h"
#include "SystemState.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// Các struct dữ liệu từ code cũ của bạn
struct RecognitionTask {
    cv::Mat frame;
    cv::Mat faceData;
};

struct UserRecord {
    std::string name;
    cv::Mat embedding;
};

struct SharedResult {
    std::mutex mtx;
    std::vector<std::pair<cv::Rect, std::string>> recognized_faces;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void updateFrame();
    void on_toggleButton_clicked();
    void on_addUserButton_clicked();
    void on_saveSettingsButton_clicked();

private:
    Ui::MainWindow *ui;
    QTimer *timer;

    cv::Ptr<cv::FaceDetectorYN> faceDetector;
    cv::Ptr<cv::FaceRecognizerSF> recognizer;
    std::vector<UserRecord> faceDatabase;

    QString botToken;
    QString chatId;

    std::atomic<bool> is_running{false};
    SharedResult sharedResult;
    SystemContext* securitySystem;

    BoundedQueue<cv::Mat>* frameQueue;
    BoundedQueue<cv::Mat>* displayQueue;
    BoundedQueue<RecognitionTask>* recognitionQueue;
    BoundedQueue<cv::Mat>* notiQueue;

    std::thread cameraThread;
    std::thread detectionThread;
    std::thread recognitionThread;
    std::thread notificationThread;

    void startAI();
    void stopAI();
    QImage cvMatToQImage(const cv::Mat& mat);
    std::string identityFace(const cv::Mat& queryEmbedding);
    void sendTelegramQt(const cv::Mat& frame);
    void loadDatabase();
    void loadConfig();
};

#endif // MAINWINDOW_H