# matrix-engine

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run Tests

```bash
./build/matrix_tests
```

## Run Benchmark

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

`multiplySequential`, `multiplyParallel`, `determinantSequential`, and `determinantParallel`
are defined in `include/ParallelOperation.hpp`.
