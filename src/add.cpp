#include "ParallelOperation.hpp"
#include "BenchmarkHelpers.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>

using namespace matrix_engine;

template<int N>
void runCase(int iters, size_t threads) {
    std::mt19937_64 rng(12345);
    Matrix<float, N> a;
    Matrix<float, N> b;
    AddOperation const addOperation{};
    benchmark::fillRandom(a, rng);
    benchmark::fillRandom(b, rng);

    benchmark::BenchmarkRunner runner(N, 4, iters, threads);
    runner.run<Matrix<float, N, N>>(
        [&]() {
            return addOperation(SeqMode{}, a, b);
        },
        [&](Executor &executor) {
            return addOperation(ParMode{executor}, a, b);
        },
        [&](Executor &executor) {
            return addOperation(ParMode{executor}, a, b);
        }
    );
}

template<int... Ns>
void runAllCases(int iters, size_t threads) {
    // Fold expression to run all cases in the parameter pack
    (runCase<Ns>(iters, threads), ...);
}


int main(int argc, char **argv) {
    int iters = 5;
    size_t threads = std::thread::hardware_concurrency();

    if (argc >= 2) {
        iters = std::max(1, std::atoi(argv[1]));
    }
    if (argc >= 3) {
        threads = static_cast<size_t>(std::max(1, std::atoi(argv[2])));
    }

    std::cout << "Benchmark add: \tSequential \tvs\t LockExecutor \tvs\t LockFreeExecutor\n";
    std::cout << "threads = " << threads << "\n";


    // 12000^2 = 144 million additions = 0.5 GB, MUST use heap-allocated matrices!
    runAllCases<100, 1000, 10000, 12000>(iters, threads);

    return 0;
}
