#include <iostream>
#include <memory>
#include <vector>
#include <cstring> 
#include <chrono>
#include <string>
#include <cuda_runtime.h> 
#include "FSST-LIKE-Matching/include/utils.hpp"

extern "C" void parse_batched(const uint8_t* buffer, const size_t* offsets, const size_t* lengths, int* results, size_t num_elements, int method, int pattern_id, size_t chunk_size);
extern "C" void init_pattern_something();
extern "C" void init_pattern_under();
extern "C" bool parse_cpu_something(const uint8_t* compressed, size_t len);
extern "C" bool parse_cpu_under(const uint8_t* compressed, size_t len);

void printUsage() {
    std::cerr << "Usage: ./benchmark <implementation> [targetFile] [pattern] [chunk_size]\n"
              << "Implementations: cpu, gpu_chunk_switch, gpu_chunk_matrix, gpu_warp_switch, gpu_warp_matrix\n"
              << "Patterns: under, something\n"
              << "Chunk Size: Integer (default 10, only used for chunking methods)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(); return 1; }

    const char* implementation = argv[1];
    const char* targetFile = (argc > 2) ? argv[2] : "PostHistory";
    const char* pattern_name = (argc > 3) ? argv[3] : "under";
    
    // Parse dynamic chunk size, defaulting to 10
    size_t chunk_size = (argc > 4) ? std::stoull(argv[4]) : 10;

    bool is_gpu = (std::strncmp(implementation, "gpu_", 4) == 0);
    bool is_cpu = (std::strcmp(implementation, "cpu") == 0);

    if (!is_gpu && !is_cpu) { printUsage(); return 1; }

    int method_id = 0;
    if (is_gpu) {
        if (std::strcmp(implementation, "gpu_chunk_switch") == 0) method_id = 1;
        else if (std::strcmp(implementation, "gpu_chunk_matrix") == 0) method_id = 2;
        else if (std::strcmp(implementation, "gpu_warp_switch") == 0) method_id = 3;
        else if (std::strcmp(implementation, "gpu_warp_matrix") == 0) method_id = 4;
    }

    std::cout << "Loading file: " << targetFile << std::endl;
    std::unique_ptr<file::FileData> dataContainer = file::readBinaryFileData(targetFile);
    if (!dataContainer) return 1;

    size_t numElements = dataContainer->getNumElements();
    int successCount = 0;
    float kernel_time_ms = 0.0f;

    auto total_start = std::chrono::high_resolution_clock::now();

    if (is_gpu) {
        std::cout << "Using GPU Batched Implementation [" << implementation << "]\n";
        if (method_id == 1 || method_id == 2) {
            std::cout << "Configured Chunk Size: " << chunk_size << std::endl;
        }
        
        size_t total_length = dataContainer->getTotalLength();
        const size_t* host_indexes = dataContainer->getIndexes();
        
        std::vector<size_t> host_lengths(numElements);
        for (size_t i = 0; i < numElements; ++i) host_lengths[i] = dataContainer->get(i).size();

        uint8_t* d_buffer; size_t* d_offsets; size_t* d_lengths; int* d_results;
        cudaMallocManaged(&d_buffer, total_length);
        cudaMallocManaged(&d_offsets, numElements * sizeof(size_t));
        cudaMallocManaged(&d_lengths, numElements * sizeof(size_t));
        cudaMallocManaged(&d_results, numElements * sizeof(int));
        cudaMemset(d_results, 0, numElements * sizeof(int));

        std::memcpy(d_buffer, dataContainer->getBuffer(), total_length);
        std::memcpy(d_offsets, host_indexes, numElements * sizeof(size_t));
        std::memcpy(d_lengths, host_lengths.data(), numElements * sizeof(size_t));

        int pattern_id = (std::strcmp(pattern_name, "something") == 0) ? 2 : 1;

        if (method_id == 2 || method_id == 4) {
            if (pattern_id == 2) init_pattern_something();
            else init_pattern_under();
        }

        int device = 0; cudaGetDevice(&device);
        cudaMemPrefetchAsync(d_buffer, total_length, device);
        cudaMemPrefetchAsync(d_offsets, numElements * sizeof(size_t), device);
        cudaMemPrefetchAsync(d_lengths, numElements * sizeof(size_t), device);
        cudaMemPrefetchAsync(d_results, numElements * sizeof(int), device);
        cudaDeviceSynchronize();

        std::cout << "Launching bulk GPU kernel..." << std::endl;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // Pass the dynamic chunk_size here
        parse_batched(d_buffer, d_offsets, d_lengths, d_results, numElements, method_id, pattern_id, chunk_size);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_time_ms, start, stop);

        for (size_t i = 0; i < numElements; ++i) if (d_results[i] == 1) successCount++;

        cudaFree(d_buffer); cudaFree(d_offsets); cudaFree(d_lengths); cudaFree(d_results);
        cudaEventDestroy(start); cudaEventDestroy(stop);

    } else {
        std::cout << "Using CPU Implementation.\nStarting the parsing benchmark..." << std::endl;
        auto cpu_compute_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < numElements; ++i) {
            auto span = dataContainer->get(i);
            bool matched = (std::strcmp(pattern_name, "something") == 0) 
                         ? parse_cpu_something(span.data(), span.size()) 
                         : parse_cpu_under(span.data(), span.size());
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
    if (is_gpu && (method_id == 1 || method_id == 2)) {
        std::cout << "Chunk Size:     " << chunk_size << std::endl;
    }
    std::cout << "Total matches: " << successCount << " / " << numElements << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Raw Compute Time:     " << kernel_time_ms << " ms" << std::endl;
    std::cout << "Total Roundtrip Time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}