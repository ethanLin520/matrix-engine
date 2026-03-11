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
            return value;
        }
    }
};

} // namespace matrix_engine

#endif // PARALLEL_OPERATION_HPP
