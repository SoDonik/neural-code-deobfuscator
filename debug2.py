import traceback
from benchmarks.run_benchmarks import run_benchmark
try:
    print("Running Level 3 directly")
    run_benchmark(3)
except Exception as e:
    traceback.print_exc()
