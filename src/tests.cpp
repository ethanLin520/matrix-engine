#include "ParallelOperation.hpp"

#include <cmath>
#include <iostream>
#include <random>

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
    matrix_engine::Executor executor(threads);
    const auto par = matrix_engine::multiplyParallel(a, b, executor);

    if (!almostEqual(seq, par)) {
        std::cerr << "Mismatch for dimensions " << R << "x" << K << " * "
                  << K << "x" << C << "\n";
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

    std::cout << "All tests passed.\n";
    return 0;
}
