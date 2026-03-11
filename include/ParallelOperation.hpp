#ifndef PARALLEL_OPERATION_HPP
#define PARALLEL_OPERATION_HPP

#include "Matrix.h"
#include "Executor.hpp"

#include <future>
#include <variant>
#include <vector>

using std::floating_point;
using std::future;
using std::vector;

namespace matrix_engine {

struct SeqMode {};

struct ParMode {
    Executor &executor;
};

using ExecutionMode = std::variant<SeqMode, ParMode>;   // Open for extension

/*
RAII guard to ensure that all pending tasks are
either completed or cancelled when the guard goes out of scope.
*/
template<typename FutureT>
class PendingTasksGuard {
public:
    PendingTasksGuard(Executor &executor, vector<FutureT> &taskQueue) noexcept
        : executor(executor), q(taskQueue) {}

    ~PendingTasksGuard() noexcept {
        if (!active) {
            return;
        }
        try {
            executor.cancel();
            for (auto &job : q) {
                if (job.valid()) {
                    job.wait();
                }
            }
        } catch (...) {
            // Never throw from cleanup in destructor.
        }
    }

    void dismiss() noexcept {
        active = false;
    }

private:
    Executor &executor;
    vector<FutureT> &q;
    bool active = true;
};

class MultiplyOperation {
public:
    template<floating_point T, int a, int b, int c>
    Matrix<T, a, c> operator()(
        const ExecutionMode &mode,
        const Matrix<T, a, b> &l,
        const Matrix<T, b, c> &r
    ) const {
        return std::visit([&](const auto &executionMode) {
            return run(executionMode, l, r);
        }, mode);
    }

private:
    template<floating_point T, int a, int b, int c>
    Matrix<T, a, c> run(
        const SeqMode &,
        const Matrix<T, a, b> &l,
        const Matrix<T, b, c> &r
    ) const {
        return l * r;
    }

    template<floating_point T, int a, int b, int c>
    Matrix<T, a, c> run(
        const ParMode &mode,
        const Matrix<T, a, b> &l,
        const Matrix<T, b, c> &r
    ) const {
        Matrix<T, a, c> result;
        vector<future<void>> jobs;
        jobs.reserve(a);
        PendingTasksGuard<future<void>> guard(mode.executor, jobs);

        // Parallelize over rows to keep each task writing to disjoint output data.
        for (int i = 0; i < a; ++i) {
            jobs.emplace_back(mode.executor.submit([&l, &r, &result, i]() {
                for (int j = 0; j < c; ++j) {
                    T total = 0;
                    for (int k = 0; k < b; ++k) {
                        total += l(i, k) * r(k, j);
                    }
                    result(i, j) = total;
                }
            }));
        }

        for (auto &job : jobs) {
            job.get();
        }
        guard.dismiss();

        return result;
    }
};

class DeterminantOperation {
public:
    template<floating_point T, int n>
    T operator()(
        const ExecutionMode &mode,
        const Matrix<T, n, n> &m
    ) const {
        return std::visit([&](const auto &executionMode) {
            return run(executionMode, m);
        }, mode);
    }

private:
    template<floating_point T, int n>
    T run(
        const SeqMode &,
        const Matrix<T, n, n> &m
    ) const {
        return m.determinant();
    }

    template<floating_point T, int n>
    T run(
        const ParMode &mode,
        const Matrix<T, n, n> &m
    ) const {
        if constexpr (n <= 3) {
            return m.determinant();
        } else {
            vector<future<T>> jobs;
            jobs.reserve(n);
            PendingTasksGuard<future<T>> guard(mode.executor, jobs);

            for (int i = 0; i < n; ++i) {
                const T coefficient = (i % 2 ? -1 : 1) * m(i, 0);
                if (coefficient == static_cast<T>(0)) {
                    continue;
                }

                const auto minorMatrix = m.minor(i, 0);
                jobs.emplace_back(mode.executor.submit([coefficient, minorMatrix]() {
                    return coefficient * minorMatrix.determinant();
                }));
            }

            T value = 0;
            for (auto &job : jobs) {
                value += job.get();
            }
            guard.dismiss();
            return value;
        }
    }
};

} // namespace matrix_engine

#endif // PARALLEL_OPERATION_HPP
