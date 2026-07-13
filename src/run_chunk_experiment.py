import subprocess
import csv
import re
import sys

# ==========================================
# Benchmark Configuration
# ==========================================
EXECUTABLE = "./benchmark"
IMPLEMENTATION = "gpu_chunk_switch"  # Change to gpu_chunk_switch if desired
DATASET = "PostHistory"
PATTERN = "something"
MAX_CHUNK_SIZE = 100
CSV_FILENAME = f"chunk_size_results_switch_{PATTERN}.csv"

# Regex to extract the timing values from your C++ console output
RAW_TIME_REGEX = re.compile(r"Raw Compute Time:\s+([0-9.]+)\s+ms")
TOTAL_TIME_REGEX = re.compile(r"Total Roundtrip Time:\s+([0-9.]+)\s+ms")

def run_benchmarks():
    print(f"Starting benchmark sweep for {IMPLEMENTATION} on pattern '{PATTERN}'")
    print(f"Testing chunk sizes from 1 to {MAX_CHUNK_SIZE}...\n")
    
    # Open CSV file for writing
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ChunkSize", "RawComputeTime_ms", "TotalRoundtripTime_ms"])
        
        for chunk_size in range(1, MAX_CHUNK_SIZE + 1):
            print(f"Running chunk size {chunk_size:3d}... ", end='', flush=True)
            
            # Build the command: ./benchmark gpu_chunk_matrix PostHistory something <chunk_size>
            cmd = [EXECUTABLE, IMPLEMENTATION, DATASET, PATTERN, str(chunk_size)]
            
            try:
                # Execute the C++ program and capture the output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout
                
                # Parse the times
                raw_match = RAW_TIME_REGEX.search(output)
                total_match = TOTAL_TIME_REGEX.search(output)
                
                if raw_match and total_match:
                    raw_time = float(raw_match.group(1))
                    total_time = float(total_match.group(1))
                    
                    # Save to CSV
                    writer.writerow([chunk_size, raw_time, total_time])
                    print(f"Raw: {raw_time:7.2f} ms | Total: {total_time:7.2f} ms")
                else:
                    print("Error: Could not parse output.")
                    
            except subprocess.CalledProcessError as e:
                print(f"Failed! Error code: {e.returncode}")
                print(e.stderr)

    print(f"\nBenchmarking complete! Results saved to {CSV_FILENAME}")

if __name__ == "__main__":
    run_benchmarks()