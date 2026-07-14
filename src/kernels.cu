#include <cuda_runtime.h>
#include <cstdint>

// State Definitions
#define Q0 0
#define Q1 1
#define Q2 2
#define Q3 3
#define Q4 4
#define Q5 5
#define Q6 6
#define Q7 7
#define Q8 8

// Accept States
#define UNDER_E1 6
#define UNDER_E2 7
#define SOMETHING_ACCEPT 9

__constant__ uint8_t d_jump_under[256];
__constant__ uint8_t d_jump_something[256];

extern "C" void init_gpu_tables() {
    uint8_t h_under[256] = {Q0}; 
    h_under[65] = Q4; 
    h_under[76] = Q2;
    
    uint8_t h_some[256] = {Q0};
    h_some[51] = Q5;   // 51 jumps directly to Q5
    uint8_t some_q4[] = {17, 21, 33, 67, 118, 132};
    for(uint8_t c : some_q4) {
        h_some[c] = Q4; 
    }
    
    cudaMemcpyToSymbol(d_jump_under, h_under, 256 * sizeof(uint8_t));
    cudaMemcpyToSymbol(d_jump_something, h_some, 256 * sizeof(uint8_t));
}

__device__ __forceinline__ int next_state_under(int state, uint8_t c) {
    switch(state) {
        case Q0: return (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
        case Q1: return (c == 8) ? UNDER_E2 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
        case Q2: return (c == 59) ? Q1 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
        case Q3: return (c == 168 || c == 156 || c == 118 || c == 87 || c == 41 || c == 28 || c == 11 || c == 6) ? UNDER_E1 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
        case Q4: return (c == 14) ? Q3 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
        case Q5: default: return Q0;
    }
}

__device__ __forceinline__ int next_state_something(int state, uint8_t c) {
    switch(state) {
        case Q0: return (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q1: return (c == 89) ? SOMETHING_ACCEPT : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q2: return (c == 79) ? Q1 : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q3: return (c == 75) ? Q2 : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q4: return (c == 167) ? Q3 : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q5: return (c == 75) ? Q6 : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q6: return (c == 79) ? Q7 : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q7: return (c == 89) ? SOMETHING_ACCEPT : (c == 51) ? Q5 : (c == 132 || c == 67 || c == 118 || c == 33 || c == 17 || c == 21) ? Q4 : (c == 255) ? Q8 : Q0;
        case Q8: default: return Q0;
    }
}

__global__ void horizontalChunkUnder(const uint8_t* buffer, const size_t* offsets, int* results, size_t num_elements, size_t chunk_size) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = offsets[element_idx + 1] - offset; 
    const uint8_t* compressed = buffer + offset;

    size_t level = 5; 

    for (size_t chunk_start = threadIdx.x * chunk_size; chunk_start < len; chunk_start += blockDim.x * chunk_size) {
        size_t end_idx = chunk_start + chunk_size + level;
        if (end_idx > len) end_idx = len;

        int state = Q0;
        for (size_t i = chunk_start; i < end_idx; ++i) {
            if (state == Q0 && i > 0 && compressed[i - 1] == 255) continue;
            state = next_state_under(state, compressed[i]);
            
            if (state == UNDER_E1 || state == UNDER_E2) { 
                results[element_idx] = 1; // Direct Write-Combining global write
                return;
            }
        }
    }
}

__global__ void horizontalChunkSomething(const uint8_t* buffer, const size_t* offsets, int* results, size_t num_elements, size_t chunk_size) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = offsets[element_idx + 1] - offset; 
    const uint8_t* compressed = buffer + offset;

    size_t level = 5; 

    for (size_t chunk_start = threadIdx.x * chunk_size; chunk_start < len; chunk_start += blockDim.x * chunk_size) {
        size_t end_idx = chunk_start + chunk_size + level;
        if (end_idx > len) end_idx = len;

        int state = Q0;
        for (size_t i = chunk_start; i < end_idx; ++i) {
            if (state == Q0 && i > 0 && compressed[i - 1] == 255) continue;
            state = next_state_something(state, compressed[i]);
            
            if (state == SOMETHING_ACCEPT) { 
                results[element_idx] = 1; // Direct Write-Combining global write
                return;
            }
        }
    }
}

__global__ void warpUnder(const uint8_t* buffer, const size_t* offsets, int* results, size_t num_elements) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = offsets[element_idx + 1] - offset; 
    const uint8_t* compressed = buffer + offset;

    for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
        if (idx > 0 && compressed[idx - 1] == 255) continue;
        
        int state = d_jump_under[compressed[idx]];
        
        if (state != Q0) {
            size_t match_offset = 1;
            
            while (state != Q0 && state != UNDER_E1 && state != UNDER_E2 && (idx + match_offset) < len) {
                state = next_state_under(state, compressed[idx + match_offset]);
                match_offset++;
            }
            if (state == UNDER_E1 || state == UNDER_E2) {
                results[element_idx] = 1; // Direct Global Write
                return; // Instantly retire thread
            }
        }
    }
}

__global__ void warpSomething(const uint8_t* buffer, const size_t* offsets, int* results, size_t num_elements) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = offsets[element_idx + 1] - offset; 
    const uint8_t* compressed = buffer + offset;

    for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
        if (idx > 0 && compressed[idx - 1] == 255) continue;
        
        int state = d_jump_something[compressed[idx]];
        
        if (state != Q0) {
            size_t match_offset = 1;
            
            while (state != Q0 && state != SOMETHING_ACCEPT && (idx + match_offset) < len) {
                state = next_state_something(state, compressed[idx + match_offset]);
                match_offset++;
            }
            if (state == SOMETHING_ACCEPT) {
                results[element_idx] = 1; // Direct Global Write
                return; // Instantly retire thread
            }
        }
    }
}

extern "C" void parse_batched(const uint8_t* buffer, const size_t* offsets, int* results, size_t num_elements, int method, int pattern_id, size_t chunk_size) {
    if (num_elements == 0) return;
    
    size_t blocksPerGrid = num_elements;
    int threadsPerBlock = 256;

    if (method == 1) { 
        // Horizontal Chunking
        if (pattern_id == 1) {
            horizontalChunkUnder<<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, results, num_elements, chunk_size);
        } else {
            horizontalChunkSomething<<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, results, num_elements, chunk_size);
        }
    }
    else if (method == 2) { 
        // Warp Speculative
        if (pattern_id == 1) {
            warpUnder<<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, results, num_elements);
        } else {
            warpSomething<<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, results, num_elements);
        }
    }
    
    cudaDeviceSynchronize();
}