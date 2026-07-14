#include <iostream>
#include <memory>
#include <vector>
#include <cstring> 
#include <chrono>
#include <string>
#include <cuda_runtime.h> 
#include "../FSST-LIKE-Matching/include/utils.hpp"

// External Linkages
extern "C" void parse_batched(const uint8_t* buffer, const size_t* offsets, int* results, size_t num_elements, int method, int pattern_id, size_t chunk_size);
extern "C" void init_gpu_tables(); 
extern "C" bool parse_cpu_something(const uint8_t* compressed, size_t len);
extern "C" bool parse_cpu_under(const uint8_t* compressed, size_t len);

void printUsage() {
    std::cerr << "Usage: ./benchmark <implementation> [targetFile] [pattern] [chunk_size]\n"
              << "Implementations: cpu, gpu_chunk, gpu_warp\n"
              << "Patterns: under, something\n"
              << "Chunk Size: Integer (default 10, only used for chunking)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { 
        printUsage(); 
        return 1; 
    }

    const char* implementation = argv[1];
    const char* targetFile = (argc > 2) ? argv[2] : "../PostHistory";
    const char* pattern_name = (argc > 3) ? argv[3] : "under";
    
    size_t chunk_size = (argc > 4) ? std::stoull(argv[4]) : 10;

    bool is_gpu = (std::strncmp(implementation, "gpu_", 4) == 0);
    bool is_cpu = (std::strcmp(implementation, "cpu") == 0);

    if (!is_gpu && !is_cpu) { 
        printUsage(); 
        return 1; 
    }

    int method_id = 0;
    if (is_gpu) {
        if (std::strcmp(implementation, "gpu_chunk") == 0) method_id = 1;
        else if (std::strcmp(implementation, "gpu_warp") == 0) method_id = 2;
    }
    
    // Pattern ID mapping: 1 = under, 2 = something
    int pattern_id = (std::strcmp(pattern_name, "something") == 0) ? 2 : 1;

    std::cout << "Loading file: " << targetFile << std::endl;
    std::unique_ptr<file::FileData> dataContainer = file::readBinaryFileData(targetFile);
    if (!dataContainer) {
        std::cerr << "Failed to load data container." << std::endl;
        return 1;
    }

    size_t numElements = dataContainer->getNumElements();

    size_t total_length = dataContainer->getTotalLength();
    const size_t* host_indexes = dataContainer->getIndexes();
    std::cout << "Number of elements: " << numElements << std::endl;
    int successCount = 0;
    float kernel_time_ms = 0.0f;

    auto total_start = std::chrono::high_resolution_clock::now();

    uint8_t* d_buffer; 
    size_t* d_offsets; 
    int* d_results;
    
    // Buffer allocation
    cudaMallocManaged(&d_buffer, total_length);
    
    // Allocate N+1 offsets to calculate lengths without a separate array
    cudaMallocManaged(&d_offsets, (numElements + 1) * sizeof(size_t));
    
    // Results is now just a single int boolean flag per string
    cudaMallocManaged(&d_results, numElements * sizeof(int));
    cudaMemset(d_results, 0, numElements * sizeof(int));

    // Copy data directly into managed memory
    std::memcpy(d_buffer, dataContainer->getBuffer(), total_length);
    std::memcpy(d_offsets, host_indexes, numElements * sizeof(size_t));
    
    // Secure the final index boundary for the N+1 math
    d_offsets[numElements] = total_length; 

    // Asynchronously prefetch pages to whichever device is active
    int device = 0; 
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(d_buffer, total_length, device);
    cudaMemPrefetchAsync(d_offsets, (numElements + 1) * sizeof(size_t), device);
    cudaMemPrefetchAsync(d_results, numElements * sizeof(int), device);
    cudaDeviceSynchronize();

    // Initialize constant memory transition tables if GPU mode is active
    if (is_gpu) {
        init_gpu_tables();
    }

    if (is_gpu) {
        std::cout << "Using GPU Batched Implementation [" << implementation << "]\n";
        if (method_id == 1) {
            std::cout << "Configured Chunk Size: " << chunk_size << std::endl;
        }
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        parse_batched(d_buffer, d_offsets, d_results, numElements, method_id, pattern_id, chunk_size);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_time_ms, start, stop);

        // Tally results
        for (size_t i = 0; i < numElements; ++i) {
            if (d_results[i] == 1) successCount++;
        }

        cudaEventDestroy(start); 
        cudaEventDestroy(stop);

    } else {
        std::cout << "Using CPU Implementation.\nStarting the parsing benchmark..." << std::endl;
        
        auto cpu_compute_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < numElements; ++i) {
            // N+1 Offset Math for dynamic length calculation
            size_t offset = d_offsets[i];
            size_t len = d_offsets[i + 1] - offset;
            const uint8_t* span = d_buffer + offset;
            
            bool matched = (pattern_id == 2) 
                         ? parse_cpu_something(span, len) 
                         : parse_cpu_under(span, len);
                         
            if (matched) successCount++;
        }
        
        auto cpu_compute_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_duration = cpu_compute_end - cpu_compute_start;
        kernel_time_ms = cpu_duration.count();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = total_end - total_start;

    std::cout << "========================================" << std::endl;
    std::cout << "Processing Complete." << std::endl;
    std::cout << "Implementation: " << implementation << std::endl;
    std::cout << "Target Pattern: " << pattern_name << std::endl;
    if (is_gpu && method_id == 1) {
        std::cout << "Chunk Size:     " << chunk_size << std::endl;
    }
    std::cout << "Total matches: " << successCount << " / " << numElements << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Raw Compute Time:     " << kernel_time_ms << " ms" << std::endl;
    std::cout << "Total Roundtrip Time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
    
    cudaFree(d_buffer); 
    cudaFree(d_offsets); 
    cudaFree(d_results);
    
    return 0;
}