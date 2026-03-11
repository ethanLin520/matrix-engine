#ifndef BENCHMARK_HELPERS_HPP
#define BENCHMARK_HELPERS_HPP

#include "Executor.hpp"
#include "Matrix.h"

#include <chrono>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <random>

namespace matrix_engine {
namespace benchmark {

template<int R, int C>
void fillRandom(Matrix<double, R, C> &m, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            m(i, j) = dist(rng);
        }
    }
}

inline double speedup(long long seqUs, long long otherUs) {
    if (seqUs <= 0 || otherUs <= 0) {
        return 0.0;
    }
    return static_cast<double>(seqUs) / static_cast<double>(otherUs);
}

inline void printResultRow(
    int n,
    int nWidth,
    int iters,
    long long seqUs,
    long long parUs,
    long long lockFreeUs
) {
    std::cout << "N=" << std::setw(nWidth) << n
              << "  iters=" << std::setw(4) << iters
              << "  seq(μs)=" << std::setw(10) << seqUs
              << "  par(μs)=" << std::setw(10) << parUs
              << "  lf(μs)=" << std::setw(10) << lockFreeUs
              << "  par_speedup = " << std::fixed << std::setprecision(2)
              << speedup(seqUs, parUs)
              << "x"
              << "  lf_speedup = "
              << speedup(seqUs, lockFreeUs)
              << "x\n";
}

template<std::floating_point T>
inline void consumeResult(T seqOut, T parOut, T lockFreeOut) {
    volatile T sink = seqOut + parOut + lockFreeOut;
    (void)sink;
}

template<std::floating_point T, int R, int C>
inline void consumeResult(
    const Matrix<T, R, C> &seqOut,
    const Matrix<T, R, C> &parOut,
    const Matrix<T, R, C> &lockFreeOut
) {
    volatile T sink = seqOut(0, 0) + parOut(0, 0) + lockFreeOut(0, 0);
    (void)sink;
}


class BenchmarkRunner {
public:
    BenchmarkRunner(
        int n,
        int nWidth,
        int iters,
        size_t threads,
        size_t lockFreeQueueCapacity = 1024
    )
        : n(n),
          nWidth(nWidth),
          iters(iters),
          threads(threads),
          lockFreeQueueCapacity(lockFreeQueueCapacity) {}

    template<typename Result, typename SeqCallable, typename ParCallable, typename LockFreeCallable>
    void run(
        SeqCallable &&seqCallable,
        ParCallable &&parCallable,
        LockFreeCallable &&lockFreeCallable
    ) const {
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::microseconds;

        auto seqStart = high_resolution_clock::now();
        Result seqOut{};
        for (int i = 0; i < iters; ++i) {
            seqOut = seqCallable();
        }
        auto seqEnd = high_resolution_clock::now();

        auto parStart = high_resolution_clock::now();
        Result parOut{};
        LockExecutor executor(threads);
        for (int i = 0; i < iters; ++i) {
            parOut = parCallable(executor);
        }
        auto parEnd = high_resolution_clock::now();

        auto lockFreeStart = high_resolution_clock::now();
        Result lockFreeOut{};
        LockFreeExecutor lockFreeExecutor(threads, lockFreeQueueCapacity);
        for (int i = 0; i < iters; ++i) {
            lockFreeOut = lockFreeCallable(lockFreeExecutor);
        }
        auto lockFreeEnd = high_resolution_clock::now();

        consumeResult(seqOut, parOut, lockFreeOut);

        const auto seqUs = duration_cast<microseconds>(seqEnd - seqStart).count();
        const auto parUs = duration_cast<microseconds>(parEnd - parStart).count();
        const auto lockFreeUs = duration_cast<microseconds>(lockFreeEnd - lockFreeStart).count();

        printResultRow(n, nWidth, iters, seqUs, parUs, lockFreeUs);
    }

private:
    int n;
    int nWidth;
    int iters;
    size_t threads;
    size_t lockFreeQueueCapacity;
};

} // namespace benchmark
} // namespace matrix_engine

#endif // BENCHMARK_HELPERS_HPP
