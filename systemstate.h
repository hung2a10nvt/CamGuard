#ifndef SYSTEMSTATE_H
#define SYSTEMSTATE_H

#include <chrono>
#include <string>
#include <memory>
#include "BoundedQueue.h"
#include <opencv2/opencv.hpp>

enum class FaceEvent {NO_FACE, OWNER_DETECTED, STRANGER_DETECTED};

class SystemContext;
class SystemState;
class SafeState;
class SusState;
class DangerState;

class SystemState {
public:
    virtual ~SystemState() = default;
    virtual void handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) = 0;
    virtual std::string getName() = 0;
};

class SystemContext {
private:
    BoundedQueue<cv::Mat>& notiQueue;
    std::unique_ptr<SystemState> currentState;
public:
    SystemContext(BoundedQueue<cv::Mat>& queue);
    void transitionTo(std::unique_ptr<SystemState> newState);
    void processEvent(FaceEvent event, const cv::Mat& frame);
    void triggerTelegram(const cv::Mat& frame);
};

class SafeState : public SystemState {
public:
    void handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) override;
    std::string getName() override { return "SAFE"; }
};

class SusState : public SystemState {
private:
    std::chrono::steady_clock::time_point start_time;
public:
    SusState();
    void handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) override;
    std::string getName() override { return "SUSPICIOUS"; }
};

class DangerState : public SystemState {
public:
    void handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) override;
    std::string getName() override { return "DANGER"; }
};

#endif // SYSTEMSTATE_H