# Matrix Engine

Matrix Engine is a C++20 fixed-size matrix library with sequential and parallel execution modes.
It includes matrix multiplication and determinant operations, with both lock-based and lock-free executors for benchmarking and experimentation.

## Requirements

- C++20-compatible compiler
- CMake 3.16+

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run Tests

```bash
./build/matrix_tests
```

## Run Code

```bash
./build/matrix_benchmark [iters] [threads]
./build/matrix_determinant_benchmark [iters] [threads]
```

Examples:

```bash
./build/matrix_benchmark 5
./build/matrix_benchmark 8 4
./build/matrix_determinant_benchmark 50
./build/matrix_determinant_benchmark 20 4
```

## Run Benchmark

```bash
cd benchmark
chmod +x benchmark.sh

./benchmark
```

## Design

Visitor-dispatched operations are defined in `include/ParallelOperation.hpp`:
- `MultiplyOperation`
- `DeterminantOperation`
- `ExecutionMode` (`SeqMode` or `ParMode{Executor&}`)

Example:

```cpp
matrix_engine::MultiplyOperation multiplyOperation;
const matrix_engine::ExecutionMode seqMode{matrix_engine::SeqMode{}};
const matrix_engine::ExecutionMode parMode{matrix_engine::ParMode{executor}};

auto seq = multiplyOperation(seqMode, a, b);
auto par = multiplyOperation(parMode, a, b);
```

Design notes:
- `ParMode` stores `Executor&`; the executor must outlive each operation call.
- `MultiplyOperation` and `DeterminantOperation` are stateless and can be reused across threads.
- Determinant parallel execution only parallelizes top-level Laplace terms (not recursive parallelism).
