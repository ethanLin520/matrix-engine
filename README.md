# Matrix Engine

Matrix Engine is a C++20 matrix computation library with sequential and parallel execution modes.

## Requirements

- C++20-compatible compiler
- CMake 3.16+

## Build

```bash
cmake -S . -B build
cmake --build build --config Release -j
```

## Run Tests

```bash
./build/matrix_tests
```

## Run Code

```bash
./build/add_benchmark [iters=5] [threads=8]
./build/multiply_benchmark [iters=5] [threads=8]
./build/determinant_benchmark [iters=10] [threads=8]
```

## Run Benchmark Script

```bash
cd benchmark
chmod +x benchmark.sh

./benchmark.sh
```
### Result
See `.txt` output files in [`/benchmark`](/benchmark). 

The benchmark was run on an Apple M3 CPU with 8 cores (4 performance and 4 efficiency). The performance will vary across machines and platforms.

## Architecture

The library is organized into three layers:

**Matrix** (`include/Matrix.h`) — Template class `Matrix<T, rows, cols>` with compile-time dimensions. Move-only semantics backed by `std::unique_ptr`. Provides element access, minor computation, determinant, and addition.

**Executors** (`include/Executor.hpp`) — Abstract `Executor` base with `submit(Func, Args...) -> std::future<R>` and `cancel()`. Two concrete implementations:
- `LockExecutor` — mutex + condition variable, tasks queued in `std::queue`
- `LockFreeExecutor` — atomic circular buffer with CAS loops and `wait`/`notify`

**Parallel Operations** (`include/ParallelOperation.hpp`) — Stateless, reusable operations dispatched over `ExecutionMode` (`SeqMode` or `ParMode{Executor&}`) via `std::visit`:
- `MultiplyOperation` / `AddOperation` — row-level parallelization; each worker thread writes to an independent row
- `DeterminantOperation` — parallelizes top-level Laplace cofactor terms only (not recursive calls)
- `PendingTasksGuard` — **RAII** guard that calls `cancel()` on exception, `dismiss()` on success

Example:

```cpp
matrix_engine::MultiplyOperation multiplyOperation;
matrix_engine::ExecutionMode const seqMode{matrix_engine::SeqMode{}};
matrix_engine::ExecutionMode const parMode{matrix_engine::ParMode{executor}};

auto seq = multiplyOperation(seqMode, a, b);
auto par = multiplyOperation(parMode, a, b);
```

## Design

### Operation

- Operations (`MultiplyOperation`, `AddOperation`, `DeterminantOperation`) are independent structs with a templated `operator()`, not subclasses of a common base.
- A virtual base would require erasing the matrix type parameters (`T`, `rows`, `cols`) behind a common interface, forcing runtime dispatch on types that are fully known at compile time.
- The template parameters differ per operation (multiply takes two differently-sized matrices; determinant takes a square matrix), so no single virtual signature could cover all three without type erasure overhead.
- Using plain structs keeps the implementation stateless, zero-overhead, and open to overloading for new type combinations without touching a class hierarchy.

### Open-Close Principle (OCP)

- Current design is OCP along the operation axis: adding a new operation (e.g. `TransposeOperation`) requires no changes to existing code. It is not OCP along the execution mode axis: adding a new mode (e.g. `GpuMode`) requires extending the `ExecutionMode` variant and adding a `run()` overload in every existing operation. This is the inherent expression problem tradeoff of `std::variant` + `std::visit` — the inverse would be true with a virtual hierarchy. The current tradeoff is intentional since execution modes are stable and new operations are the more likely extension point.

### Executor Lifetime

- `ParMode` stores `Executor&`; the executor must outlive each operation call.
- `ParMode` borrows an executor instead of owning it via `std::unique_ptr` or `std::shared_ptr`.
- Borrowing keeps ownership explicit and avoids hidden lifetime extension.
- `std::unique_ptr` would imply ownership transfer, which is not desired for a reusable thread pool.
- `std::shared_ptr` would add shared-ownership overhead and make shutdown timing less predictable.
- Operations submit jobs and wait before returning, so a non-owning reference is sufficient.

### Executor Reusability

- Executors are designed to be reusable.
- `cancel()` is queue-drain behavior: it removes pending tasks but does not stop tasks already running.
- `PendingTasksGuard` is an RAII guard that calls `cancel()` on early exit (for example, exceptions); normal completion calls `dismiss()`.

### Matrix Data

- Matrix data is stored as `std::unique_ptr<array<array<T, cols>, rows>>` rather than a plain member array.
- A plain `array<array<T, cols>, rows>` member would be allocated inline. For large matrices (e.g. N=12000 as used in benchmarks), this exceeds typical stack limits (~8 MB) — a 12000×12000 `float` matrix is ~576 MB — making heap allocation a hard requirement.
- Heap allocation via `unique_ptr` also makes moves O(1) — only the pointer is transferred — which is important since operations like `minor()` and `determinant()` return matrices by value through deep recursion.
- Copy constructor and copy assignment are explicitly deleted to enforce move-only semantics and prevent accidental deep copies.

### Pending Tasks Guard

- Each parallel operation submits tasks to the executor and holds a `vector<future<T>>` of in-flight jobs.
- If an exception is thrown mid-operation (e.g. during task submission), futures already in the vector would be abandoned — their results never waited on — while the executor continues running the associated tasks against now-dangling references.
- `PendingTasksGuard` wraps the executor and the job vector. Its destructor calls `executor.cancel()` to drain the pending queue, then waits on any already-running futures, preventing use-after-free on captured references.
- On the success path, `dismiss()` deactivates the guard so the destructor is a no-op.
- The guard is non-copyable and non-movable to prevent accidental transfer of this responsibility.
