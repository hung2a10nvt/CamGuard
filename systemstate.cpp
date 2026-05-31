#include "SystemState.h"

SystemContext::SystemContext(BoundedQueue<cv::Mat>& queue) : notiQueue(queue) {
    currentState = std::make_unique<SafeState>();
}

void SystemContext::transitionTo(std::unique_ptr<SystemState> newState) {
    currentState = std::move(newState);
    std::cout << "New state: " << currentState->getName() << "\n";
}

void SystemContext::processEvent(FaceEvent event, const cv::Mat& frame) {
    currentState->handle(this, event, frame);
}

void SystemContext::triggerTelegram(const cv::Mat& frame) {
    notiQueue.push(frame.clone());
}


void SafeState::handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) {
    if (event == FaceEvent::STRANGER_DETECTED) {
        context->transitionTo(std::make_unique<SusState>());
    }
}

SusState::SusState() {
    start_time = std::chrono::steady_clock::now();
}

void SusState::handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) {
    if (event == FaceEvent::NO_FACE){
        std::cout << "Stranger left, back to safe.\n";
        context->transitionTo(std::make_unique<SafeState>());
        return;
    }

    if (event == FaceEvent::OWNER_DETECTED) {
        std::cout << "Owner detected, no problem.\n";
        context->transitionTo(std::make_unique<SafeState>());
        return;
    }

    if (event == FaceEvent::STRANGER_DETECTED) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (elapsed >= 3) {
            context->transitionTo(std::make_unique<DangerState>());
            context->triggerTelegram(frame);
        }
    }
}

void DangerState::handle(SystemContext* context, FaceEvent event, const cv::Mat& frame) {
    if (event == FaceEvent::OWNER_DETECTED) {
        std::cout << "Owner's back. Disarming alarm.\n";
        context->transitionTo(std::make_unique<SafeState>());
    }
}