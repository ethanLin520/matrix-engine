#include "ParallelOperation.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

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

template<typename T, typename U>
void requireAlmostEqual(const T &actual, const U &expected, const std::string &message) {
    if (!almostEqual(actual, expected)) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

template<typename Exception, typename Fn>
void requireThrows(Fn &&fn, const std::string &message) {
    try {
        fn();
    } catch (const Exception &) {
        return;
    }
    std::cerr << message << "\n";
    std::exit(1);
}

template<int R, int C>
void fillRandomMatrix(matrix_engine::Matrix<double, R, C> &m, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            m(i, j) = dist(rng);
        }
    }
}

template<int R, int K, int C>
void checkRandomCase(size_t threads, std::mt19937_64 &rng) {
    const matrix_engine::MultiplyOperation multiplyOperation{};

    matrix_engine::Matrix<double, R, K> a;
    matrix_engine::Matrix<double, K, C> b;

    fillRandomMatrix(a, rng);
    fillRandomMatrix(b, rng);

    const auto seq = multiplyOperation(matrix_engine::SeqMode{}, a, b);
    matrix_engine::LockExecutor executor(threads);
    const auto par = multiplyOperation(matrix_engine::ParMode{executor}, a, b);
    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(R));
    const auto lockFreePar = multiplyOperation(matrix_engine::ParMode{lockFreeExecutor}, a, b);

    requireAlmostEqual(
        seq,
        par,
        "Mismatch for dimensions " + std::to_string(R) + "x" + std::to_string(K) + " * " +
            std::to_string(K) + "x" + std::to_string(C) + " (LockExecutor)"
    );

    requireAlmostEqual(
        seq,
        lockFreePar,
        "Mismatch for dimensions " + std::to_string(R) + "x" + std::to_string(K) + " * " +
            std::to_string(K) + "x" + std::to_string(C) + " (LockFreeExecutor)"
    );
}

template<int N>
void checkDeterminantKnownCase(
    const matrix_engine::Matrix<double, N, N> &m,
    double expected,
    size_t threads
) {
    const matrix_engine::DeterminantOperation determinantOperation{};
    const auto seq = determinantOperation(matrix_engine::SeqMode{}, m);
    requireAlmostEqual(
        seq,
        expected,
        "Sequential determinant mismatch for " + std::to_string(N) + "x" +
            std::to_string(N) + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(seq)
    );

    matrix_engine::LockExecutor executor(threads);
    const auto par = determinantOperation(matrix_engine::ParMode{executor}, m);
    requireAlmostEqual(
        par,
        expected,
        "LockExecutor determinant mismatch for " + std::to_string(N) + "x" +
            std::to_string(N) + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(par)
    );

    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(N));
    const auto lockFreePar = determinantOperation(matrix_engine::ParMode{lockFreeExecutor}, m);
    requireAlmostEqual(
        lockFreePar,
        expected,
        "LockFreeExecutor determinant mismatch for " + std::to_string(N) + "x" +
            std::to_string(N) + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(lockFreePar)
    );
}

template<int N>
void checkRandomDeterminantCase(size_t threads, std::mt19937_64 &rng) {
    const matrix_engine::DeterminantOperation determinantOperation{};

    matrix_engine::Matrix<double, N, N> m;
    fillRandomMatrix(m, rng);

    const auto seq = determinantOperation(matrix_engine::SeqMode{}, m);

    matrix_engine::LockExecutor executor(threads);
    const auto par = determinantOperation(matrix_engine::ParMode{executor}, m);
    requireAlmostEqual(
        seq,
        par,
        "Random determinant mismatch for " + std::to_string(N) + "x" + std::to_string(N) +
            " (LockExecutor)"
    );

    matrix_engine::LockFreeExecutor lockFreeExecutor(threads, 4 * static_cast<size_t>(N));
    const auto lockFreePar = determinantOperation(matrix_engine::ParMode{lockFreeExecutor}, m);
    requireAlmostEqual(
        seq,
        lockFreePar,
        "Random determinant mismatch for " + std::to_string(N) + "x" + std::to_string(N) +
            " (LockFreeExecutor)"
    );
}

void checkMatrixBoundsAndInitSafety() {
    requireThrows<std::invalid_argument>(
        [] {
            matrix_engine::Matrix<double, 2, 2> badCols{{1.0, 2.0, 3.0}};
            (void)badCols;
        },
        "Expected invalid_argument for too many columns in initializer"
    );

    requireThrows<std::invalid_argument>(
        [] {
            matrix_engine::Matrix<double, 2, 2> badRows{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
            };
            (void)badRows;
        },
        "Expected invalid_argument for too many rows in initializer"
    );

    matrix_engine::Matrix<double, 2, 2> m{{1.0, 2.0}, {3.0, 4.0}};
    requireThrows<std::out_of_range>(
        [&m] { (void)m(2, 0); },
        "Expected out_of_range for non-const index access"
    );

    const matrix_engine::Matrix<double, 2, 2> cm{{1.0, 2.0}, {3.0, 4.0}};
    requireThrows<std::out_of_range>(
        [&cm] { (void)cm(0, 2); },
        "Expected out_of_range for const index access"
    );
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
