
echo "Running benchmarks..."
echo "Multiply Benchmark Results\n" > multiply_benchmark_results.txt
echo "Determinant Benchmark Results\n" > determinant_benchmark_results.txt
echo "Add Benchmark Results\n" > add_benchmark_results.txt

for thread in 1 2 4 8; do
    echo "Running with $thread threads..."

    echo "\n" >> multiply_benchmark_results.txt
    ../build/multiply_benchmark 5 $thread >> multiply_benchmark_results.txt

    echo "\n" >> determinant_benchmark_results.txt
    ../build/determinant_benchmark 20 $thread >> determinant_benchmark_results.txt

    ../build/add_benchmark 1 > /dev/null # warm up the cache
    echo "\n" >> add_benchmark_results.txt
    ../build/add_benchmark 5 $thread >> add_benchmark_results.txt
done


echo "Done"

