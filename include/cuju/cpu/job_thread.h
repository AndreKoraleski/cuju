// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// A thread that runs one job at a time while its caller goes on, for an asynchronous step.
#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "../floating_point.h"

namespace cuju {
namespace cpu {

class JobThread {
public:
    JobThread() = default;
    JobThread(const JobThread &) = delete;
    JobThread &operator=(const JobThread &) = delete;
    ~JobThread() { Stop(); }

    // Runs `job` on the thread, which starts with `setup` at the first job. Throws
    // `std::logic_error` while another job runs.
    void Launch(std::function<void()> job, const std::function<void()> &setup) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (pending_) throw std::logic_error("expected a wait for the step in flight first");
        if (!thread_.joinable()) thread_ = std::thread([this, setup] { Work(setup); });
        job_ = std::move(job);
        failure_ = nullptr;
        pending_ = true;
        launched_.store(true, std::memory_order_release);
        lock.unlock();
        wake_.notify_all();
    }

    // Waits for the job, then throws what it threw, once.
    void Wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return !pending_; });
        if (failure_) {
            std::exception_ptr failure = failure_;
            failure_ = nullptr;
            std::rethrow_exception(failure);
        }
    }

    // Waits for the job without its exception, unless the caller is the thread itself. Every other
    // call on the buffers waits for the step that writes them.
    void Settle() {
        // Speed: a batch that never stepped asynchronously takes no lock.
        if (!launched_.load(std::memory_order_acquire)) return;
        std::unique_lock<std::mutex> lock(mutex_);
        if (std::this_thread::get_id() == threadId_) return;
        done_.wait(lock, [this] { return !pending_; });
    }

    // Whether a job was in flight, read without the lock by a forked child, whose only thread is
    // the caller.
    bool Pending() const { return pending_; }

    void Stop() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!thread_.joinable()) return;
            done_.wait(lock, [this] { return !pending_; });
            stop_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

private:
    void Work(const std::function<void()> &setup) {
        // Exact: the thread computes in the default floating point environment.
        DefaultFloatingPoint floatingPoint;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            threadId_ = std::this_thread::get_id();
        }
        setup();
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || (pending_ && job_); });
                if (stop_) return;
                job = std::move(job_);
                job_ = nullptr;
            }
            std::exception_ptr failure;
            try {
                job();
            } catch (...) {
                failure = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                failure_ = failure;
                pending_ = false;
            }
            done_.notify_all();
        }
    }

    std::thread thread_;
    std::thread::id threadId_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::function<void()> job_;
    std::exception_ptr failure_;
    bool pending_ = false, stop_ = false;
    std::atomic<bool> launched_{false};
};

}  // namespace cpu
}  // namespace cuju
