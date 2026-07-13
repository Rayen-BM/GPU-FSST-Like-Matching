#include <cuda_runtime.h>
#include <cstdint>

#define Q0 0
#define Q1 1
#define Q2 2
#define Q3 3
#define Q4 4
#define Q5 5
#define UNDER_E1 6
#define UNDER_E2 7
#define Q6 6
#define Q7 7
#define Q8 8
#define SOMETHING_ACCEPT 9

#define PATTERN_UNDER 1
#define PATTERN_SOMETHING 2

#define MAX_STATES 10

__constant__ uint8_t d_transition_matrix[MAX_STATES * 256];

extern "C" void upload_transition_matrix(const uint8_t* host_matrix) {
    cudaMemcpyToSymbol(d_transition_matrix, host_matrix, MAX_STATES * 256);
}

// -------------------------------------------------------------
// Hardcoded Transition Logic
// -------------------------------------------------------------
template <int PATTERN_ID>
__device__ __forceinline__ int next_state_switch(int state, uint8_t c) {
    if constexpr (PATTERN_ID == PATTERN_UNDER) {
        switch(state) {
            case Q0: return (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
            case Q1: return (c == 8) ? UNDER_E2 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
            case Q2: return (c == 59) ? Q1 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
            case Q3: return (c == 168 || c == 156 || c == 118 || c == 87 || c == 41 || c == 28 || c == 11 || c == 6) ? UNDER_E1 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
            case Q4: return (c == 14) ? Q3 : (c == 65) ? Q4 : (c == 76) ? Q2 : (c == 255) ? Q5 : Q0;
            case Q5: default: return Q0;
        }
    } else if constexpr (PATTERN_ID == PATTERN_SOMETHING) {
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
    return Q0;
}

// -------------------------------------------------------------
// Method 1: Chunking Switch
// -------------------------------------------------------------
template <int PATTERN_ID>
__global__ void batched_chunking_switch(const uint8_t* buffer, const size_t* offsets, const size_t* lengths, int* results, size_t num_elements, size_t chunk_size) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = lengths[element_idx];
    const uint8_t* compressed = buffer + offset;

    __shared__ int found;
    if (threadIdx.x == 0) found = 0;
    __syncthreads();

    constexpr size_t max_pattern_len = (PATTERN_ID == PATTERN_UNDER) ? 7 : 11;

    for (size_t chunk_start = threadIdx.x * chunk_size; chunk_start < len && !found; chunk_start += blockDim.x * chunk_size) {
        size_t end_idx = chunk_start + chunk_size + (max_pattern_len - 1);
        if (end_idx > len) end_idx = len;

        int state = Q0;
        for (size_t i = chunk_start; i < end_idx; ++i) {
            if (state == Q0 && i > 0 && compressed[i - 1] == 255) continue;
            state = next_state_switch<PATTERN_ID>(state, compressed[i]);
            
            if constexpr (PATTERN_ID == PATTERN_UNDER) {
                if (state == UNDER_E1 || state == UNDER_E2) { atomicExch(&found, 1); break; }
            } else {
                if (state == SOMETHING_ACCEPT) { atomicExch(&found, 1); break; }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && found) results[element_idx] = 1;
}

// -------------------------------------------------------------
// Method 2: Chunking Matrix
// -------------------------------------------------------------
template <int PATTERN_ID>
__global__ void batched_chunking_matrix(const uint8_t* buffer, const size_t* offsets, const size_t* lengths, int* results, size_t num_elements, size_t chunk_size) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = lengths[element_idx];
    const uint8_t* compressed = buffer + offset;

    __shared__ int found;
    if (threadIdx.x == 0) found = 0;
    __syncthreads();

    constexpr size_t max_pattern_len = (PATTERN_ID == PATTERN_UNDER) ? 7 : 11;

    for (size_t chunk_start = threadIdx.x * chunk_size; chunk_start < len && !found; chunk_start += blockDim.x * chunk_size) {
        size_t end_idx = chunk_start + chunk_size + (max_pattern_len - 1);
        if (end_idx > len) end_idx = len;

        int state = Q0;
        for (size_t i = chunk_start; i < end_idx; ++i) {
            if (state == Q0 && i > 0 && compressed[i - 1] == 255) continue;
            state = d_transition_matrix[(state << 8) | compressed[i]];
            
            if constexpr (PATTERN_ID == PATTERN_UNDER) {
                if (state == UNDER_E1 || state == UNDER_E2) { atomicExch(&found, 1); break; }
            } else {
                if (state == SOMETHING_ACCEPT) { atomicExch(&found, 1); break; }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && found) results[element_idx] = 1;
}

// -------------------------------------------------------------
// Method 3: Warp Switch
// -------------------------------------------------------------
template <int PATTERN_ID>
__global__ void batched_warp_switch(const uint8_t* buffer, const size_t* offsets, const size_t* lengths, int* results, size_t num_elements) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = lengths[element_idx];
    const uint8_t* compressed = buffer + offset;

    __shared__ int found;
    if (threadIdx.x == 0) found = 0;
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < len && !found; idx += blockDim.x) {
        if (idx > 0 && compressed[idx - 1] == 255) continue;
        int state = next_state_switch<PATTERN_ID>(Q0, compressed[idx]);
        if (state != Q0) {
            size_t match_offset = 1;
            if constexpr (PATTERN_ID == PATTERN_UNDER) {
                while (state != Q0 && state != UNDER_E1 && state != UNDER_E2 && (idx + match_offset) < len) {
                    state = next_state_switch<PATTERN_ID>(state, compressed[idx + match_offset]);
                    match_offset++;
                }
                if (state == UNDER_E1 || state == UNDER_E2) atomicExch(&found, 1);
            } else {
                while (state != Q0 && state != SOMETHING_ACCEPT && (idx + match_offset) < len) {
                    state = next_state_switch<PATTERN_ID>(state, compressed[idx + match_offset]);
                    match_offset++;
                }
                if (state == SOMETHING_ACCEPT) atomicExch(&found, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && found) results[element_idx] = 1;
}

// -------------------------------------------------------------
// Method 4: Warp Matrix
// -------------------------------------------------------------
template <int PATTERN_ID>
__global__ void batched_warp_matrix(const uint8_t* buffer, const size_t* offsets, const size_t* lengths, int* results, size_t num_elements) {
    size_t element_idx = blockIdx.x;
    if (element_idx >= num_elements) return;

    size_t offset = offsets[element_idx];
    size_t len = lengths[element_idx];
    const uint8_t* compressed = buffer + offset;

    __shared__ int found;
    if (threadIdx.x == 0) found = 0;
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < len && !found; idx += blockDim.x) {
        if (idx > 0 && compressed[idx - 1] == 255) continue;
        int state = d_transition_matrix[compressed[idx]];
        if (state != Q0) {
            size_t match_offset = 1;
            if constexpr (PATTERN_ID == PATTERN_UNDER) {
                while (state != Q0 && state != UNDER_E1 && state != UNDER_E2 && (idx + match_offset) < len) {
                    state = d_transition_matrix[(state << 8) | compressed[idx + match_offset]];
                    match_offset++;
                }
                if (state == UNDER_E1 || state == UNDER_E2) atomicExch(&found, 1);
            } else {
                while (state != Q0 && state != SOMETHING_ACCEPT && (idx + match_offset) < len) {
                    state = d_transition_matrix[(state << 8) | compressed[idx + match_offset]];
                    match_offset++;
                }
                if (state == SOMETHING_ACCEPT) atomicExch(&found, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && found) results[element_idx] = 1;
}

// -------------------------------------------------------------
// Wrapper Entry Point
// -------------------------------------------------------------
extern "C" void parse_batched(const uint8_t* buffer, const size_t* offsets, const size_t* lengths, int* results, size_t num_elements, int method, int pattern_id, size_t chunk_size) {
    if (num_elements == 0) return;
    size_t blocksPerGrid = num_elements;
    int threadsPerBlock = 256;

    if (method == 1) {
        if (pattern_id == PATTERN_UNDER) batched_chunking_switch<PATTERN_UNDER><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements, chunk_size);
        else batched_chunking_switch<PATTERN_SOMETHING><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements, chunk_size);
    }
    else if (method == 2) {
        if (pattern_id == PATTERN_UNDER) batched_chunking_matrix<PATTERN_UNDER><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements, chunk_size);
        else batched_chunking_matrix<PATTERN_SOMETHING><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements, chunk_size);
    }
    else if (method == 3) {
        if (pattern_id == PATTERN_UNDER) batched_warp_switch<PATTERN_UNDER><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements);
        else batched_warp_switch<PATTERN_SOMETHING><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements);
    }
    else if (method == 4) {
        if (pattern_id == PATTERN_UNDER) batched_warp_matrix<PATTERN_UNDER><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements);
        else batched_warp_matrix<PATTERN_SOMETHING><<<blocksPerGrid, threadsPerBlock>>>(buffer, offsets, lengths, results, num_elements);
    }
    
    cudaDeviceSynchronize();
}