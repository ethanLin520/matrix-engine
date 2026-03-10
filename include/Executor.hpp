#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "Matrix.h"

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>


using std::vector;
using std::queue;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::function;
using std::shared_ptr;
using std::packaged_task;
using std::future;
using std::make_shared;

namespace matrix_engine {

class Executor {
public:
    Executor(
        size_t num_threads = thread::hardware_concurrency()
    ) : num_threads(num_threads) { start(); };

    ~Executor() noexcept{
        {
            unique_lock<mutex> lock(mtx);
            shutdown = true;
        }
        cv.notify_all();
        for (auto &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) {
        using R = std::invoke_result_t<Func, Args...>;
        auto task = make_shared<packaged_task<R()>>(
            [func = std::forward<Func>(func), ... args = std::forward<Args>(args)]() mutable {
                return func(args...);
            }
        );
        future<R> res = task->get_future();

        auto success = enqueueTask([task]() {
            (*task)();
        }); // lambda wrapper to execute the task

        if (!success) {
            throw std::runtime_error("Failed to submit task");
        }

        return res;
    }

private:

    bool enqueueTask(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (shutdown)
                return false; // Reject new tasks if shutting down

            tasks.push(std::move(job));
        }
        cv.notify_one();
        return true;
    }


    void worker_thread() {
        while (true) {
            // Wait for tasks or shutdown signal
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [this] { return !tasks.empty() || shutdown; });

            if (shutdown && tasks.empty()) {
                break;
            }

            // Get the next task
            auto t = tasks.front();
            tasks.pop();
            lock.unlock();

            // Execute the task
            t();
        }
    }

    void start() {
        // Clamp num_threads to at least 1
        num_threads = num_threads > 0 ? num_threads : 1;

        workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            // constructs thread (this->worker_thread) inplace
            workers.emplace_back(&Executor::worker_thread, this);
        }
    }

private:
    size_t num_threads = thread::hardware_concurrency();
    vector<thread> workers;
    queue<function<void()>> tasks;

    // Sync
    mutex mtx;
    condition_variable cv;
    bool shutdown = false;

public:
    // Delete copy and move constructors and assignment operators

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    Executor(Executor&&) = delete;
    Executor& operator=(Executor&&) = delete;

};


class LockFreeExecutor {
public:
    LockFreeExecutor(
        size_t num_threads = thread::hardware_concurrency(),
        size_t queue_capacity = 1024
    )
        : num_threads(num_threads > 0 ? num_threads : 1),
          queue_capacity(queue_capacity > 0 ? queue_capacity : 1),
          slots(std::make_unique<TaskSlot[]>(this->queue_capacity))
    {
        work_signal.clear(std::memory_order_relaxed);
        start();
    }

    ~LockFreeExecutor() noexcept {
        shutdown.store(true, std::memory_order_release);
        work_signal.test_and_set(std::memory_order_release);
        work_signal.notify_all();
        for (auto &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) {
        using R = std::invoke_result_t<Func, Args...>;
        auto task = make_shared<packaged_task<R()>>(
            [func = std::forward<Func>(func), ... args = std::forward<Args>(args)]() mutable {
                return func(args...);
            }
        );
        future<R> res = task->get_future();

        auto success = enqueueTask([task]() {
            (*task)();
        });

        if (!success) {
            throw std::runtime_error("Failed to submit task");
        }

        return res;
    }

public:
    LockFreeExecutor(const LockFreeExecutor&) = delete;
    LockFreeExecutor& operator=(const LockFreeExecutor&) = delete;
    LockFreeExecutor(LockFreeExecutor&&) = delete;
    LockFreeExecutor& operator=(LockFreeExecutor&&) = delete;

private:
    struct TaskSlot {
        std::atomic_flag ready = ATOMIC_FLAG_INIT;
        function<void()> fn;

        TaskSlot() {
            ready.clear(std::memory_order_relaxed);
        }
    };

    bool enqueueTask(function<void()> job) {
        while (true) {
            if (shutdown.load(std::memory_order_acquire)) {
                return false;
            }

            size_t t = tail.load(std::memory_order_relaxed);
            const size_t h = head.load(std::memory_order_acquire);
            if (t - h >= queue_capacity) {
                std::this_thread::yield();
                continue;
            }

            if (!tail.compare_exchange_weak(
                    t, t + 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                continue;
            }

            TaskSlot &slot = slots[t % queue_capacity];
            while (slot.ready.test(std::memory_order_acquire)) {
                slot.ready.wait(true, std::memory_order_relaxed);
            }

            slot.fn = std::move(job);
            slot.ready.test_and_set(std::memory_order_release);
            slot.ready.notify_one();

            pending.fetch_add(1, std::memory_order_release);
            work_signal.test_and_set(std::memory_order_release);
            work_signal.notify_one();
            return true;
        }
    }

    bool tryDequeueTask(function<void()> &out) {
        while (true) {
            size_t h = head.load(std::memory_order_relaxed);
            const size_t t = tail.load(std::memory_order_acquire);
            if (h >= t) {
                return false;
            }

            if (!head.compare_exchange_weak(
                    h, h + 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                continue;
            }

            TaskSlot &slot = slots[h % queue_capacity];
            while (!slot.ready.test(std::memory_order_acquire)) {
                slot.ready.wait(false, std::memory_order_relaxed);
            }

            out = std::move(slot.fn);
            slot.fn = {};
            slot.ready.clear(std::memory_order_release);
            slot.ready.notify_one();
            return true;
        }
    }

    void worker_thread() {
        while (true) {
            function<void()> job;
            if (tryDequeueTask(job)) {
                job();
                const size_t left = pending.fetch_sub(1, std::memory_order_acq_rel) - 1;
                if (left == 0) {
                    work_signal.clear(std::memory_order_release);
                    if (pending.load(std::memory_order_acquire) > 0) {
                        work_signal.test_and_set(std::memory_order_release);
                        work_signal.notify_one();
                    }
                }
                continue;
            }

            if (shutdown.load(std::memory_order_acquire) &&
                pending.load(std::memory_order_acquire) == 0) {
                break;
            }

            if (pending.load(std::memory_order_acquire) == 0) {
                work_signal.clear(std::memory_order_release);
            }
            work_signal.wait(false, std::memory_order_relaxed);
        }
    }

    void start() {
        workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back(&LockFreeExecutor::worker_thread, this);
        }
    }

private:
    size_t num_threads;
    size_t queue_capacity;
    vector<thread> workers;
    std::unique_ptr<TaskSlot[]> slots;

    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};
    std::atomic<size_t> pending{0};

    std::atomic<bool> shutdown{false};
    std::atomic_flag work_signal = ATOMIC_FLAG_INIT;
};

} // namespace matrix_engine

#endif // EXECUTOR_H
