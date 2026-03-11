#include "ParallelOperation.hpp"
#include "BenchmarkHelpers.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>

using namespace matrix_engine;

template<int N>
void runDeterminantCase(int iters, size_t threads) {
    std::mt19937_64 rng(54321);
    Matrix<double, N> m;
    DeterminantOperation const determinantOp{};
    benchmark::fillRandom(m, rng);

    benchmark::BenchmarkRunner runner(
        N, 2, iters, threads, 4 * static_cast<size_t>(N)
    );
    runner.run<double>(
        [&]() {
            return determinantOp(SeqMode{}, m);
        },
        [&](Executor &executor) {
            return determinantOp(ParMode{executor}, m);
        },
        [&](Executor &executor) {
            return determinantOp(ParMode{executor}, m);
        }
    );
}

template<int... Ns>
void runAllDeterminantCases(int iters, size_t threads) {
    (runDeterminantCase<Ns>(iters, threads), ...);
}


int main(int argc, char **argv) {
    int iters = 10;
    size_t threads = std::thread::hardware_concurrency();

    if (argc >= 2) {
        iters = std::max(1, std::atoi(argv[1]));
    }
    if (argc >= 3) {
        threads = static_cast<size_t>(std::max(1, std::atoi(argv[2])));
    }

    std::cout << "Benchmark determinant:\tSequential \tvs\t LockExecutor \tvs\t LockFreeExecutor\n";
    std::cout << "threads = " << threads << "\n";

    // Laplace expansion grows factorially, so benchmark only small square matrices.
    runAllDeterminantCases<4, 6, 8, 10, 12>(iters, threads);

    return 0;
}
