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

bool almostEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

template<int R, int K, int C>
void checkRandomCase(size_t threads, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    const matrix_engine::MultiplyOperation multiplyOperation{};

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

    const auto seq = multiplyOperation(matrix_engine::SeqMode{}, a, b);
    matrix_engine::LockExecutor executor(threads);
    const auto par = multiplyOperation(matrix_engine::ParMode{executor}, a, b);
    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(R));
    const auto lockFreePar = multiplyOperation(matrix_engine::ParMode{lockFreeExecutor}, a, b);

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

template<int N>
void checkDeterminantKnownCase(
    const matrix_engine::Matrix<double, N, N> &m,
    double expected,
    size_t threads
) {
    const matrix_engine::DeterminantOperation determinantOperation{};
    const auto seq = determinantOperation(matrix_engine::SeqMode{}, m);
    if (!almostEqual(seq, expected)) {
        std::cerr << "Sequential determinant mismatch for " << N << "x" << N
                  << ": expected " << expected << ", got " << seq << "\n";
        std::exit(1);
    }

    matrix_engine::LockExecutor executor(threads);
    const auto par = determinantOperation(matrix_engine::ParMode{executor}, m);
    if (!almostEqual(par, expected)) {
        std::cerr << "LockExecutor determinant mismatch for " << N << "x" << N
                  << ": expected " << expected << ", got " << par << "\n";
        std::exit(1);
    }

    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(N));
    const auto lockFreePar = determinantOperation(matrix_engine::ParMode{lockFreeExecutor}, m);
    if (!almostEqual(lockFreePar, expected)) {
        std::cerr << "LockFreeExecutor determinant mismatch for " << N << "x" << N
                  << ": expected " << expected << ", got " << lockFreePar << "\n";
        std::exit(1);
    }
}

template<int N>
void checkRandomDeterminantCase(size_t threads, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    const matrix_engine::DeterminantOperation determinantOperation{};

    matrix_engine::Matrix<double, N, N> m;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            m(i, j) = dist(rng);
        }
    }

    const auto seq = determinantOperation(matrix_engine::SeqMode{}, m);

    matrix_engine::LockExecutor executor(threads);
    const auto par = determinantOperation(matrix_engine::ParMode{executor}, m);
    if (!almostEqual(seq, par)) {
        std::cerr << "Random determinant mismatch for " << N << "x" << N
                  << " (LockExecutor)\n";
        std::exit(1);
    }

    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(N));
    const auto lockFreePar = determinantOperation(matrix_engine::ParMode{lockFreeExecutor}, m);
    if (!almostEqual(seq, lockFreePar)) {
        std::cerr << "Random determinant mismatch for " << N << "x" << N
                  << " (LockFreeExecutor)\n";
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
    checkDeterminantKnownCase<1>({{5.5}}, 5.5, threads);
    checkDeterminantKnownCase<2>({{1.0, 2.0}, {3.0, 4.0}}, -2.0, threads);
    checkDeterminantKnownCase<3>(
        {{6.0, 1.0, 1.0}, {4.0, -2.0, 5.0}, {2.0, 8.0, 7.0}},
        -306.0,
        threads
    );
    checkRandomDeterminantCase<4>(threads, rng);
    checkRandomDeterminantCase<5>(threads, rng);
    checkMatrixBoundsAndInitSafety();

    std::cout << "All tests passed.\n";
    return 0;
}
