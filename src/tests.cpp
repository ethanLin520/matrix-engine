#include "ParallelOperation.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

template<int R, int C>
bool almostEqual(const matrix_engine::Matrix<double, R, C> &a,
                 const matrix_engine::Matrix<double, R, C> &b,
                 double eps = 1e-9) {
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (std::abs(a(i, j) - b(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

template<int R, int K, int C>
void checkRandomCase(size_t threads, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-3.0, 3.0);

    matrix_engine::Matrix<double, R, K> a;
    matrix_engine::Matrix<double, K, C> b;

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < K; ++j) {
            a(i, j) = dist(rng);
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < C; ++j) {
            b(i, j) = dist(rng);
        }
    }

    const auto seq = matrix_engine::multiplySequential(a, b);
    matrix_engine::LockExecutor executor(threads);
    const auto par = matrix_engine::multiplyParallel(a, b, executor);
    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(R));
    const auto lockFreePar = matrix_engine::multiplyParallel(a, b, lockFreeExecutor);

    if (!almostEqual(seq, par)) {
        std::cerr << "Mismatch for dimensions " << R << "x" << K << " * "
                  << K << "x" << C << " (LockExecutor)\n";
        std::exit(1);
    }

    if (!almostEqual(seq, lockFreePar)) {
        std::cerr << "Mismatch for dimensions " << R << "x" << K << " * "
                  << K << "x" << C << " (LockFreeExecutor)\n";
        std::exit(1);
    }
}

void checkMatrixBoundsAndInitSafety() {
    bool threw = false;
    try {
        matrix_engine::Matrix<double, 2, 2> badCols{{1.0, 2.0, 3.0}};
        (void)badCols;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "Expected invalid_argument for too many columns in initializer\n";
        std::exit(1);
    }

    threw = false;
    try {
        matrix_engine::Matrix<double, 2, 2> badRows{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        };
        (void)badRows;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "Expected invalid_argument for too many rows in initializer\n";
        std::exit(1);
    }

    matrix_engine::Matrix<double, 2, 2> m{{1.0, 2.0}, {3.0, 4.0}};
    threw = false;
    try {
        (void)m(2, 0);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "Expected out_of_range for non-const index access\n";
        std::exit(1);
    }

    const matrix_engine::Matrix<double, 2, 2> cm{{1.0, 2.0}, {3.0, 4.0}};
    threw = false;
    try {
        (void)cm(0, 2);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "Expected out_of_range for const index access\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    std::mt19937_64 rng(20260310);
    const size_t threads = 4;

    checkRandomCase<1, 1, 1>(threads, rng);
    checkRandomCase<2, 3, 4>(threads, rng);
    checkRandomCase<8, 8, 8>(threads, rng);
    checkRandomCase<16, 12, 10>(threads, rng);
    checkRandomCase<32, 32, 32>(threads, rng);
    checkMatrixBoundsAndInitSafety();

    std::cout << "All tests passed.\n";
    return 0;
}
