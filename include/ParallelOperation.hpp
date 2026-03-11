#ifndef PARALLEL_OPERATION_HPP
#define PARALLEL_OPERATION_HPP

#include "Matrix.h"
#include "Executor.hpp"

#include <future>
#include <variant>
#include <vector>
#include <cmath>

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

public:
    PendingTasksGuard(PendingTasksGuard const&) = delete;
    PendingTasksGuard& operator=(PendingTasksGuard const&) = delete;
    PendingTasksGuard(PendingTasksGuard&&) = delete;
    PendingTasksGuard& operator=(PendingTasksGuard&&) = delete;
};

class MultiplyOperation {
public:
    template<floating_point T, int a, int b, int c>
    Matrix<T, a, c> operator()(
        ExecutionMode const &mode,
        Matrix<T, a, b> const &l,
        Matrix<T, b, c> const &r
    ) const {
        return std::visit([&](auto const &executionMode) {
            return run(executionMode, l, r);
        }, mode);
    }

private:
    template<floating_point T, int a, int b, int c>
    Matrix<T, a, c> run(
        SeqMode const &,
        Matrix<T, a, b> const &l,
        Matrix<T, b, c> const &r
    ) const {
        return l * r;
    }

    template<floating_point T, int a, int b, int c>
    Matrix<T, a, c> run(
        ParMode const &mode,
        Matrix<T, a, b> const &l,
        Matrix<T, b, c> const &r
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
        ExecutionMode const &mode,
        Matrix<T, n, n> const &m
    ) const {
        return std::visit([&](auto const &executionMode) {
            return run(executionMode, m);
        }, mode);
    }

private:
    template<floating_point T, int n>
    T run(
        SeqMode const &,
        Matrix<T, n, n> const &m
    ) const {
        return m.determinant();
    }

    template<floating_point T, int n>
    T run(
        ParMode const &mode,
        Matrix<T, n, n> const &m
    ) const {
        if constexpr (n <= 3) {
            return m.determinant();
        } else {
            vector<future<T>> jobs;
            jobs.reserve(n);
            PendingTasksGuard<future<T>> guard(mode.executor, jobs);

            for (int i = 0; i < n; ++i) {
                T const coefficient = (i % 2 ? -1 : 1) * m(i, 0);
                if (is_zero(coefficient)) {
                    continue;
                }

                jobs.emplace_back(mode.executor.submit([coefficient, &m, i]() {
                    return coefficient * m.minor(i, 0).determinant();
                }));
            }

            T value{0.0};
            for (auto &job : jobs) {
                value += job.get();
            }
            guard.dismiss();
            return value;
        }
    }

    template<floating_point T>
    bool is_zero(T value) const {
        // Never use == on floating point
        return std::abs(value) < static_cast<T>(1e-9);
    }
};



class AddOperation {
public:
    template<floating_point T, int a, int b>
    Matrix<T, a, b> operator()(
        ExecutionMode const &mode,
        Matrix<T, a, b> const &l,
        Matrix<T, a, b> const &r
    ) const {
        return std::visit([&](auto const &executionMode) {
            return run(executionMode, l, r);
        }, mode);
    }

private:
    template<floating_point T, int a, int b>
    Matrix<T, a, b> run(
        SeqMode const &,
        Matrix<T, a, b> const &l,
        Matrix<T, a, b> const &r
    ) const {
        return l + r;
    }

    template<floating_point T, int a, int b>
    Matrix<T, a, b> run(
        ParMode const &mode,
        Matrix<T, a, b> const &l,
        Matrix<T, a, b> const &r
    ) const {
        Matrix<T, a, b> result;
        vector<future<void>> jobs;
        jobs.reserve(a);
        PendingTasksGuard<future<void>> guard(mode.executor, jobs);

        // Parallelize over rows to keep each task writing to disjoint output data.
        for (int i = 0; i < a; ++i) {
            jobs.emplace_back(mode.executor.submit([&l, &r, &result, i]() {
                for (int j = 0; j < b; ++j) {
                    result(i, j) = l(i, j) + r(i, j);
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

} // namespace matrix_engine

#endif // PARALLEL_OPERATION_HPP
