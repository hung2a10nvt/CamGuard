#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "registerdialog.h"
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttpMultiPart>
#include <QEventLoop>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    loadConfig();
    loadDatabase();

    faceDetector = cv::FaceDetectorYN::create("face_detection_yunet_2022mar.onnx", "", cv::Size(320, 320), 0.6f, 0.3f, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);
    recognizer = cv::FaceRecognizerSF::create("face_recognition_sface_2021dec.onnx", "");

    notiQueue = new BoundedQueue<cv::Mat>(5);
    securitySystem = new SystemContext(*notiQueue);
    frameQueue = new BoundedQueue<cv::Mat>(2);
    displayQueue = new BoundedQueue<cv::Mat>(2);
    recognitionQueue = new BoundedQueue<RecognitionTask>(2);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);
}

MainWindow::~MainWindow()
{
    stopAI();
    delete timer;
    delete securitySystem;
    delete notiQueue;
    delete frameQueue;
    delete displayQueue;
    delete recognitionQueue;
    delete ui;
}


void MainWindow::on_toggleButton_clicked()
{
    if (!is_running) {
        ui->toggleButton->setText("STOP");
        ui->toggleButton->setStyleSheet("background-color: #dc3545; color: white; border-radius: 8px; font-weight: bold;");
        startAI();
        timer->start(33);
    } else {
        ui->toggleButton->setText("START");
        ui->toggleButton->setStyleSheet("background-color: #007bff; color: white; border-radius: 8px; font-weight: bold;");
        timer->stop();
        stopAI();
        ui->cameraLabel->clear();
        ui->cameraLabel->setText("Camera");
    }
}

void MainWindow::updateFrame()
{
    cv::Mat frame;
    if (displayQueue->pop(frame)) {
        QImage qImg = cvMatToQImage(frame);
        ui->cameraLabel->setPixmap(QPixmap::fromImage(qImg).scaled(ui->cameraLabel->size(), Qt::KeepAspectRatio));
    }
}

void MainWindow::on_saveSettingsButton_clicked()
{
    botToken = ui->botTokenInput->text();
    chatId = ui->chatIdInput->text();

    cv::FileStorage fs("config.xml", cv::FileStorage::WRITE);
    fs << "bot_token" << botToken.toStdString();
    fs << "chat_id" << chatId.toStdString();
    fs.release();

    QMessageBox::information(this, "Success!", "Saved Telegrams config!");
}

void MainWindow::on_addUserButton_clicked()
{
    if (is_running) {
        QMessageBox::warning(this, "Camera's not available", "Stop security before adding new user to White list!");
        return;
    }
    RegisterDialog dialog(this);
    if (dialog.exec() == QDialog::Accepted && dialog.isSuccess) {
        UserRecord newUser;
        newUser.name = dialog.newUserName;
        newUser.embedding = dialog.newUserEmbedding.clone();
        faceDatabase.push_back(newUser);

        ui->listWidget->addItem(QString::fromStdString(newUser.name));

        cv::FileStorage fs("face_db.xml", cv::FileStorage::WRITE);
        fs << "total_users" << (int)faceDatabase.size();
        for (size_t i = 0; i < faceDatabase.size(); i++) {
            fs << "name_" + std::to_string(i) << faceDatabase[i].name;
            fs << "embedding_" + std::to_string(i) << faceDatabase[i].embedding;
        }
        fs.release();
    }
}

void MainWindow::startAI()
{
    is_running = true;
    frameQueue->open();
    displayQueue->open();
    recognitionQueue->open();
    notiQueue->open();

    // Camera Thread
    cameraThread = std::thread([this](){
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cv::Mat tmp_frame;
        while (is_running) {
            cap >> tmp_frame;
            if (tmp_frame.empty()) break;
            frameQueue->push(tmp_frame.clone());
        }
    });

    // Detection Thread
    detectionThread = std::thread([this](){
        cv::Mat frame;
        cv::Size currentSize;
        while (frameQueue->pop(frame)) {
            if (currentSize != frame.size()) {
                currentSize = frame.size();
                faceDetector->setInputSize(currentSize);
            }

            cv::Mat faces;
            faceDetector->detect(frame, faces);

            RecognitionTask task;
            task.frame = frame.clone();
            if (!faces.empty()) task.faceData = faces.clone();
            recognitionQueue->push(std::move(task));

            std::vector<std::pair<cv::Rect, std::string>> localStorage;
            {
                std::unique_lock<std::mutex> lock(sharedResult.mtx);
                localStorage = sharedResult.recognized_faces;
            }

            if (!faces.empty()) {
                for (int i = 0; i < faces.rows; i++) {
                    int x = int(faces.at<float>(i, 0));
                    int y = int(faces.at<float>(i, 1));
                    int w = int(faces.at<float>(i, 2));
                    int h = int(faces.at<float>(i, 3));
                    float conf = faces.at<float>(i, 14);

                    if (conf > 0.6) {
                        cv::Rect cur_box(x, y, w, h);
                        cv::rectangle(frame, cur_box, cv::Scalar(0, 255, 0), 2);
                        std::string label = "Detecting...";
                        cv::Point cur_center(x + w / 2, y + h / 2);

                        for (const auto& cache : localStorage) {
                            cv::Rect expanded(cache.first.x - 40, cache.first.y - 40, cache.first.width + 80, cache.first.height + 80);
                            if (expanded.contains(cur_center)) {
                                label = cache.second;
                                break;
                            }
                        }
                        cv::putText(frame, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
            displayQueue->push(std::move(frame));
        }
    });

    // Recognition Thread
    recognitionThread = std::thread([this](){
        RecognitionTask task;
        while (recognitionQueue->pop(task)) {
            if (task.faceData.empty()) {
                {
                    std::unique_lock<std::mutex> lock(sharedResult.mtx);
                    sharedResult.recognized_faces.clear();
                }
                securitySystem->processEvent(FaceEvent::NO_FACE, task.frame);
                continue;
            }

            std::vector<std::pair<cv::Rect, std::string>> current_results;
            bool hasOwner = false;
            bool hasStranger = false;

            for (int i = 0; i < task.faceData.rows; i++) {
                float conf = task.faceData.at<float>(i, 14);
                if (conf > 0.6) {
                    cv::Mat aligned_face, embedding;
                    recognizer->alignCrop(task.frame, task.faceData.row(i), aligned_face);
                    recognizer->feature(aligned_face, embedding);

                    std::string personName = identityFace(embedding);
                    if (personName == "Stranger") hasStranger = true;
                    else hasOwner = true;

                    current_results.push_back({cv::Rect(int(task.faceData.at<float>(i, 0)), int(task.faceData.at<float>(i, 1)), int(task.faceData.at<float>(i, 2)), int(task.faceData.at<float>(i, 3))), personName});
                }
            }

            {
                std::unique_lock<std::mutex> lock(sharedResult.mtx);
                sharedResult.recognized_faces = std::move(current_results);
            }

            FaceEvent finalEvent = FaceEvent::NO_FACE;
            if (hasOwner) finalEvent = FaceEvent::OWNER_DETECTED;
            else if (hasStranger) finalEvent = FaceEvent::STRANGER_DETECTED;
            securitySystem->processEvent(finalEvent, task.frame);
        }
    });

    // Notification Thread
    notificationThread = std::thread([this](){
        cv::Mat alertFrame;
        while (is_running && notiQueue->pop(alertFrame)) {
            sendTelegramQt(alertFrame);

            for (int i = 0; i < 5; ++i) {
                if (!is_running) break;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    });
}

void MainWindow::stopAI()
{
    is_running = false;
    frameQueue->close();
    displayQueue->close();
    recognitionQueue->close();
    notiQueue->close();

    if(cameraThread.joinable()) cameraThread.join();
    if(detectionThread.joinable()) detectionThread.join();
    if(recognitionThread.joinable()) recognitionThread.join();
    if(notificationThread.joinable()) notificationThread.join();
}

// ================= HELPER FUNCTIONS =================

QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return img.copy();
}

std::string MainWindow::identityFace(const cv::Mat& queryEmbedding) {
    if (faceDatabase.empty()) return "Unknown";
    std::string bestMatchName = "Unknown";
    double maxCosineScore = -1.0;
    for (const auto& user : faceDatabase) {
        double score = recognizer->match(queryEmbedding, user.embedding, cv::FaceRecognizerSF::FR_COSINE);
        if (score > maxCosineScore) {
            bestMatchName = user.name;
            maxCosineScore = score;
        }
    }
    if (maxCosineScore > 0.363f) return bestMatchName;
    return "Stranger";
}

void MainWindow::sendTelegramQt(const cv::Mat& frame) {
    // cv::Mat -> JPEG
    std::vector<uchar> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
    cv::imencode(".jpg", frame, buffer, params);

    QByteArray imageData(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    // Data form
    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    // chat_id
    QHttpPart chatPart;
    chatPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"chat_id\""));
    chatPart.setBody(chatId.toUtf8());
    multiPart->append(chatPart);

    // caption
    QHttpPart captionPart;
    captionPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"caption\""));
    captionPart.setBody(QString("Stranger things detected!").toUtf8());
    multiPart->append(captionPart);

    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"photo\"; filename=\"alert.jpg\""));
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    imagePart.setBody(imageData);
    multiPart->append(imagePart);

    QNetworkAccessManager manager;
    QUrl url("https://api.telegram.org/bot" + botToken + "/sendPhoto");
    QNetworkRequest request(url);

    QSslConfiguration conf = request.sslConfiguration();
    conf.setPeerVerifyMode(QSslSocket::VerifyNone);
    request.setSslConfiguration(conf);

    std::cout << "[TELEGRAM] Sending notification...\n";
    QNetworkReply *reply = manager.post(request, multiPart);
    multiPart->setParent(reply);

    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    if (reply->error() == QNetworkReply::NoError) {
        std::cout << "[TELEGRAM] Noti sent!\n";
    } else {
        std::cerr << "[TELEGRAM ERROR] " << reply->errorString().toStdString() << "\n";
    }

    delete reply;
}

void MainWindow::loadConfig() {
    cv::FileStorage fs("config.xml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        std::string t, i;
        fs["bot_token"] >> t;
        fs["chat_id"] >> i;
        botToken = QString::fromStdString(t);
        chatId = QString::fromStdString(i);
        ui->botTokenInput->setText(botToken);
        ui->chatIdInput->setText(chatId);
        fs.release();
    }
}

void MainWindow::loadDatabase() {
    cv::FileStorage fs("face_db.xml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        int totalUser;
        fs["total_users"] >> totalUser;
        for (int i = 0; i < totalUser; i++) {
            UserRecord user;
            fs["name_" + std::to_string(i)] >> user.name;
            fs["embedding_" + std::to_string(i)] >> user.embedding;
            faceDatabase.push_back(user);
            ui->listWidget->addItem(QString::fromStdString(user.name));
        }
        fs.release();
    }
}