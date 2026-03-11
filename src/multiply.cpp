#include "ParallelOperation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

namespace matrix_engine {

template<int R, int C>
void fillRandom(Matrix<double, R, C> &m, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            m(i, j) = dist(rng);
        }
    }
}

template<int N>
void runCase(int iters, size_t threads) {
    std::mt19937_64 rng(12345);
    Matrix<double, N> a;
    Matrix<double, N> b;
    const MultiplyOperation multiplyOperation{};
    fillRandom(a, rng);
    fillRandom(b, rng);

    auto seqStart = high_resolution_clock::now();
    Matrix<double, N, N> seqOut;
    for (int i = 0; i < iters; ++i) {
        seqOut = multiplyOperation(SeqMode{}, a, b);
    }
    auto seqEnd = high_resolution_clock::now();

    auto parStart = high_resolution_clock::now();
    Matrix<double, N, N> parOut;
    LockExecutor executor(threads);
    for (int i = 0; i < iters; ++i) {
        parOut = multiplyOperation(ParMode{executor}, a, b);
    }
    auto parEnd = high_resolution_clock::now();

    auto lockFreeStart = high_resolution_clock::now();
    Matrix<double, N, N> lockFreeOut;
    LockFreeExecutor lockFreeExecutor(threads);
    for (int i = 0; i < iters; ++i) {
        lockFreeOut = multiplyOperation(ParMode{lockFreeExecutor}, a, b);
    }
    auto lockFreeEnd = high_resolution_clock::now();

    volatile double sink = seqOut(0, 0) + parOut(0, 0) + lockFreeOut(0, 0);
    (void)sink;

    const auto seqUs = duration_cast<microseconds>(seqEnd - seqStart).count();
    const auto parUs = duration_cast<microseconds>(parEnd - parStart).count();
    const auto lockFreeUs = duration_cast<microseconds>(lockFreeEnd - lockFreeStart).count();

    std::cout << "N=" << std::setw(4) << N
              << "  iters=" << std::setw(4) << iters
              << "  seq(μs)=" << std::setw(10) << seqUs
              << "  par(μs)=" << std::setw(10) << parUs
              << "  lf(μs)=" << std::setw(10) << lockFreeUs
              << "  par_speedup=" << std::fixed << std::setprecision(2)
              << static_cast<double>(seqUs) / static_cast<double>(parUs)
              << "x"
              << "  lf_speedup="
              << static_cast<double>(seqUs) / static_cast<double>(lockFreeUs)
              << "x\n";
}

template<int... Ns>
void runAllCases(int iters, size_t threads) {
    // Fold expression to run all cases in the parameter pack
    (runCase<Ns>(iters, threads), ...);
}

} // namespace matrix_engine

int main(int argc, char **argv) {
    int iters = 5;
    size_t threads = std::thread::hardware_concurrency();

    if (argc >= 2) {
        iters = std::max(1, std::atoi(argv[1]));
    }
    if (argc >= 3) {
        threads = static_cast<size_t>(std::max(1, std::atoi(argv[2])));
    }

    std::cout << "Benchmark multiply: \tSequential \tvs\t LockExecutor \tvs\t LockFreeExecutor\n";
    std::cout << "threads = " << threads << "\n";


    matrix_engine::runAllCases<64, 128, 256, 512>(iters, threads);

    return 0;
}
