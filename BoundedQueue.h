#ifndef BOUNDEDQUEUE_H
#define BOUNDEDQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class BoundedQueue{
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_empty_;
    size_t capacity_;
    bool isClosed_ = false;

public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity){}

    void push(T&& item){
        std::unique_lock<std::mutex> lock(mutex_);

        // If the system already closed, do not receive anything
        if (isClosed_){
            return;
        }

        if (queue_.size() >= capacity_){
            queue_.pop();
        }

        queue_.push(std::move(item));
        cv_empty_.notify_one();
    }

    bool pop(T& item){
        std::unique_lock<std::mutex> lock(mutex_);
        cv_empty_.wait(lock, [this]() {return !queue_.empty() || isClosed_;});

        if (queue_.empty() && isClosed_){
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();

        return true;
    }

    void close(){
        std::unique_lock<std::mutex> lock(mutex_);
        isClosed_ = true;
        cv_empty_.notify_all();
    }

    void open(){
        std::unique_lock<std::mutex> lock(mutex_);
        isClosed_ = false;

        std::queue<T> empty_queue;
        std::swap(queue_, empty_queue);
    }
};

#endif // BOUNDEDQUEUE_H
