#ifndef PARALLEL_OPERATION_HPP
#define PARALLEL_OPERATION_HPP

#include "Matrix.h"
#include "Executor.hpp"

#include <vector>
#include <future>

using std::floating_point;
using std::future;
using std::vector;

namespace matrix_engine {

template<floating_point T, int a, int b, int c>
Matrix<T, a, c> multiplySequential(
    const Matrix<T, a, b> &l,
    const Matrix<T, b, c> &r
) {
    return l * r;
}

template<floating_point T, int a, int b, int c>
Matrix<T, a, c> multiplyParallel(
    const Matrix<T, a, b> &l,
    const Matrix<T, b, c> &r,
    Executor &executor
) {
    Matrix<T, a, c> result;
    vector<future<void>> jobs;
    jobs.reserve(a);

    // Parallelize over rows to keep each task writing to disjoint output data.
    for (int i = 0; i < a; ++i) {
        jobs.emplace_back(executor.submit([&l, &r, &result, i]() {
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

template<floating_point T, int n>
T determinantSequential(const Matrix<T, n, n> &m) {
    return m.determinant();
}

template<floating_point T, int n>
T determinantParallel(
    const Matrix<T, n, n> &m,
    Executor &executor
) {
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
            jobs.emplace_back(executor.submit([coefficient, minorMatrix]() {
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

} // namespace matrix_engine

#endif // PARALLEL_OPERATION_HPP
