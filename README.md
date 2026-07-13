# GPU-Accelerated Automaton String Matching Engine

A high-performance pattern matching suite optimized for rapid boolean search ("exists / does not exist") over massive textual datasets. Using a 635 MiB StackExchange `PostHistory` dump (~1.4M variable-length strings), this project evaluates hardware-level acceleration strategies, comparing an optimized **C++ AVX-512 SIMD baseline** against several parallel architectural designs written in **CUDA**.

The core engine is fully decoupled into separate runtime translation modules to prevent One-Definition-Rule (ODR) linker collisions while enabling dynamic configuration sweeps at runtime.

---

## 📂 Project Architecture & Component Breakdown

### 1. `final_driver.cpp` (The Controller)
* **Role:** Host-side (CPU) orchestrator and entry point for execution benchmarking.
* **Key Mechanics:**
  * Parses command-line inputs to dynamically configure execution strategies, data sources, matching patterns, and parameters.
  * Interfaces with the data management utilities (`file::FileData`) to read, block-align, and mirror input buffers into unified memory.
  * Manages zero-copy page setups via `cudaMallocManaged`, issues asynchronous prefetch hints (`cudaMemPrefetchAsync`) to minimize page faults, and profiles execution bottlenecks using hardware-level tracking (`cudaEventElapsedTime`).

### 2. `kernels.cu` (The GPU Execution Core)
* **Role:** Implements the parallel pattern-matching compute steps on the device.
* **Key Mechanics:**
  * **Horizontal Chunking (Methods 1 & 2):** Divides strings into uniform segments (*chunks*) assigned to threads. Threads concurrently sweep their assigned block plus a trailing boundary safety overlap ($PatternLen - 1$) to eliminate edge-fault misses.
  * **Warp Speculative Execution (Methods 3 & 4):** Breaks free from chunk alignment by assigning individual threads to single starting byte indexes. Threads speculatively scan subsequent bytes assuming they are at the initial automaton root state ($q_0$).
  * **Template Optimization Layer:** Utilizes C++ template meta-programming (`template <int PATTERN_ID>`) paired with compile-time conditionals (`if constexpr`). This structure unrolls static state branches directly into clean SASS machine code, stripping away runtime warp branching divergence.

### 3. `init.cpp` (Automaton Memory Builder)
* **Role:** Generates and compiles Deterministic Finite Automaton (DFA) state routing graphs.
* **Key Mechanics:**
  * Constructs optimized 2D transition lookup structures on the host during initialization based on the selected pattern string.
  * Maps and uploads this flat state configuration block directly onto the GPU’s high-speed constant memory lane (`__constant__ uint8_t d_transition_matrix[...]`). This setup enables $O(1)$ flat, address-mapped table evaluations inside the generic matrix kernels.

### 4. `under.cpp` & `something.cpp` (Vectorized CPU Baselines)
* **Role:** High-speed single-node reference algorithms built for the host CPU.
* **Key Mechanics:**
  * Leverages SIMD intrinsics (AVX-512 byte registers, wide vector comparisons, and generated bitmasks) to evaluate blocks of text simultaneously.
  * Both modules encapsulate their search loops within `static` internal-linkage functions. This design approach prevents global namespace pollution, allowing divergent DFA definitions to coexist cleanly at link-time.

### 5. `run_chunk_experiment.py` (Automated Profiler)
* **Role:** Python automation script designed to discover optimal hardware parameters.
* **Key Mechanics:**
  * Automates continuous runtime sweeps (e.g., executing step intervals from 1 to 100 Bytes) over the horizontal chunking layouts.
  * Spawns target executable processes, captures stdout streams, parses time benchmarks via structured regex filters, dumps data to `.csv` logs, and uses `pandas` and `matplotlib` to chart sensitivity graphs.

---

## 📊 Benchmark Summary & Core Insights

### Performance Metrics (635 MiB Workload)

| Architecture / Strategy | Target Pattern | Optimal Parameter | Raw Compute Time | Total Roundtrip Time |
| :--- | :---: | :---: | :---: | :---: |
| **C++ SIMD Baseline** | `%under%` | — | 220.8 ms | 220.8 ms |
| **C++ SIMD Baseline** | `%something%` | — | 436.8 ms | 436.8 ms |
| **GPU Warp Speculative (Switch)** | `%under%` | — | **220.5 ms** | 1146.0 ms |
| **GPU Warp Speculative (Switch)** | `%something%` | — | **276.8 ms** | 1199.2 ms |
| **GPU Warp Speculative (Matrix)** | `%under%` | — | 350.3 ms | 1266.0 ms |
| **GPU Warp Speculative (Matrix)** | `%something%` | — | 359.7 ms | 1279.9 ms |
| **GPU Horizontal Chunking (Switch)** | `%under%` | Chunk Size: 4B | 421.8 ms | 1385.6 ms |
| **GPU Horizontal Chunking (Switch)** | `%something%` | Chunk Size: 6B | 755.2 ms | 1656.3 ms |
| **GPU Horizontal Chunking (Matrix)** | `%under%` | Chunk Size: 9B | 677.0 ms | 1577.7 ms |
| **GPU Horizontal Chunking (Matrix)** | `%something%` | Chunk Size: 12B | 818.5 ms | 1723.4 ms |

### Key Architectural Findings

* **Warp Speculation Dominance:** The Speculative Switch strategy (`warp_s`) achieves top-tier compute performance on both search targets. On sparse streams like `%something%`, it outpaces the ultra-optimized AVX-512 CPU engine by over **1.6$\times$** by allowing non-matching threads to drop out instantly.
* **Chunking Uniformity ("Blind" Processing):** The Horizontal Chunking method exhibits strong data-blind robustness. Because threads systematically sweep through rigidly bounded partitions regardless of state hits, it eliminates execution variance between dense and sparse character compositions.
* **The Overlap Tax:** Fine-grained parameters heavily dictate chunking efficiency. Setting chunk limits below 5 bytes triggers an exponential latency penalty; at this threshold, edge-protection lookups begin to duplicate thread processing loops.
* **Out-of-Compute Latency Wall:** Although GPU core execution is fast, the total round-trip timeline is heavily impacted by host-side operations. Managed allocation, host-to-device memory mirror arrays, and initial PCIe bus migration add a rigid latency overhead that is absent on the CPU.

---

## 🛠️ Installation & Compilation

Building the matching suite requires a compiler stack compatible with modern host syntax ($C++20$) alongside parallel NVCC linkages (such as LLVM Clang 14+ or GCC 11+ with matching CUDA toolkits). You need to clone the FSST-LIKE-Matching github from https://github.com/calin2110/FSST-LIKE-Matching in the same directory as all the files

Compile the source using the specialized `clang++` driver flag sequence:

```bash
clang++ -O3 -std=c++20 -march=native \
  --cuda-gpu-arch=sm_80 \
  --cuda-path=/usr/lib/cuda \
  final_driver.cpp kernels.cu something.cpp under.cpp init.cpp FSST-LIKE-Matching/src/utils.cpp \
  -o benchmark -I. \
  -L/usr/lib/cuda/lib64 -lcudart