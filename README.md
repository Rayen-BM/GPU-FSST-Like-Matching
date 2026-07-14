# GPU-Accelerated Automaton String Matching Engine

A high-performance pattern matching suite optimized for rapid boolean search ("exists / does not exist") over massive textual datasets. Using a 635 MiB StackExchange `PostHistory` dump (~1.4M variable-length strings), this project evaluates hardware-level acceleration strategies, comparing an optimized **C++ AVX-512 SIMD baseline** against parallel architectural designs written in **CUDA**.

The core engine is fully decoupled into separate runtime translation modules to prevent One-Definition-Rule (ODR) linker collisions while enabling dynamic configuration sweeps at runtime.

---

## 📂 Project Architecture & Component Breakdown

### 1. `final_driver.cpp` (The Controller)
* **Role:** Host-side (CPU) orchestrator and entry point for execution benchmarking.
* **Key Mechanics:**
  * Parses command-line inputs to dynamically configure execution strategies, data sources, matching patterns, and parameters.
  * Utilizes an $N+1$ offset array architecture to compute variable string lengths dynamically, stripping unnecessary memory allocations.
  * Manages zero-copy page setups via `cudaMallocManaged` and issues asynchronous prefetch hints (`cudaMemPrefetchAsync`) to optimize unified memory paging. 

### 2. `kernels.cu` (The GPU Execution Core)
* **Role:** Implements the parallel pattern-matching compute steps on the device utilizing state-machine logic.
* **Key Mechanics:**
  * **No-Sync Benign Data Race Architecture:** Completely eliminates block-level synchronization (`__syncthreads()`) and `__shared__` memory bottlenecks. Threads perform direct Write-Combining global writes to report matches and execute immediate `return` statements to retire instantly, maximizing hardware scheduler efficiency.
  * **$O(1)$ State Jumping:** Evaluates the first byte using an ultra-fast `__constant__` memory lookup table (`d_jump`), allowing threads to bypass sequential evaluation logic and jump directly to the target DFA state.
  * **Horizontal Chunking:** Threads concurrently sweep assigned static blocks plus a hardcoded boundary safety overlap (`level = 5`) to eliminate edge-fault misses without the register pressure of dynamic boundary tracking.
  * **Warp Speculative Execution:** Breaks free from chunk alignment by assigning individual threads to single starting byte indexes, dropping out invalid state traces instantly.

### 3. `under.cpp` & `something.cpp` (Vectorized CPU Baselines)
* **Role:** High-speed single-node reference algorithms built for the host CPU.
* **Key Mechanics:**
  * Leverages SIMD intrinsics (AVX-512 byte registers, wide vector comparisons, and generated bitmasks) to evaluate blocks of text simultaneously.
  * Both modules encapsulate their search loops within `static` internal-linkage functions. This design approach prevents global namespace pollution, allowing divergent DFA definitions to coexist cleanly at link-time.

### 4. `run_chunk_experiment.py` (Automated Profiler)
* **Role:** Python automation script designed to discover optimal hardware parameters.
* **Key Mechanics:**
  * Automates continuous runtime sweeps (e.g., executing step intervals from 1 to 100 Bytes) over the horizontal chunking layouts.
  * Spawns target executable processes, captures stdout streams, parses time benchmarks via structured regex filters, dumps data to `.csv` logs, and uses `pandas` and `matplotlib` to chart sensitivity graphs.

---

## 📊 Benchmark Summary & Core Insights

### Performance Metrics (635 MiB Workload)

| Architecture / Strategy | Target Pattern | Optimal Parameter | Raw Compute Time | Total Roundtrip Time |
| :--- | :---: | :---: | :---: | :---: |
| **C++ SIMD Baseline** | `%under%` | — | 603.1 ms | 1436.8 ms |
| **C++ SIMD Baseline** | `%something%` | — | 943.5 ms | 3194.2 ms |
| **GPU Warp Speculative** | `%under%` | — | 321.6 ms | 1160.2 ms |
| **GPU Warp Speculative** | `%something%` | — | **329.0 ms** | **2576.1 ms** |
| **GPU Horizontal Chunking** | `%under%` | Chunk Size: 4B | **306.8 ms** | **2536.5 ms** |
| **GPU Horizontal Chunking** | `%something%` | Chunk Size: 6B | 439.4 ms | 3027.8 ms |

---

## 🛠️ Installation & Compilation

Building the matching suite requires a compiler stack compatible with modern host syntax ($C++20$) alongside parallel NVCC linkages (such as LLVM Clang 14+ or GCC 11+ with matching CUDA toolkits). You need to clone the FSST-LIKE-Matching repository from `https://github.com/calin2110/FSST-LIKE-Matching` into the same directory as the this repository.

Compile the source using the specialized `clang++` driver flag sequence:

```bash
clang++ -O3 -std=c++20 -march=native \
  --cuda-gpu-arch=sm_80 \
  --cuda-path=/usr/lib/cuda \
  src/final_driver.cpp src/kernels.cu src/something.cpp src/under.cpp ../FSST-LIKE-Matching/src/utils.cpp \
  -o benchmark -I. \
  -L/usr/lib/cuda/lib64 -lcudart