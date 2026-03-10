#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "Matrix.h"

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <atomic>


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
using std::atomic;

namespace matrix_engine {

class Executor {
public:
    Executor() { start(); };
    Executor(size_t num_threads) : num_threads(num_threads) { start(); };
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

} // namespace matrix_engine

#endif // EXECUTOR_H
