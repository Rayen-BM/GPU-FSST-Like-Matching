#include <cstddef>
#include <cstdint>
#include <bit>
#include <string_view>
#include <nmmintrin.h>
#include <pmmintrin.h>
constexpr bool transitionArrayq0[256] = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
constexpr uint8_t byte76Symbolq0 = 76;
__m128i byte76Vectorq0 = _mm_set1_epi8(*reinterpret_cast<const char*>(&byte76Symbolq0));
constexpr uint8_t byte65Symbolq0 = 65;
__m128i byte65Vectorq0 = _mm_set1_epi8(*reinterpret_cast<const char*>(&byte65Symbolq0));

enum class FwdState {
    E1,
    E2,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
};

static bool parseFwdState(const uint8_t* compressed, size_t len, int64_t* strIdx, uint8_t* symIdx){
    *strIdx = 0;
    auto q = FwdState::q0;
    uint64_t level = 3;
    while (*strIdx + level <= len) {
        switch(q) {
            case FwdState::E1:
                *symIdx = 1;
                *strIdx -= 1;
                return true;
            case FwdState::E2:
                *symIdx = 2;
                *strIdx -= 1;
                return true;
            case FwdState::q0:
            {
                int64_t maxIdx = len - level;
                uint8_t prevByte;
                int64_t currentStrIdx = *strIdx;
                while (currentStrIdx + 15 <= maxIdx) {
                    __m128i compressedVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compressed + currentStrIdx));
                    uint16_t mask = static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(byte76Vectorq0, compressedVec))) | static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(byte65Vectorq0, compressedVec)));
                    while (mask != 0) {
                        int index = std::countr_zero(mask);
                        size_t currentIndex = currentStrIdx + index;
                        if (currentIndex == 0 || compressed[currentIndex - 1] != 255) {
                            currentStrIdx = currentIndex;
                            goto q0;
                        }
                        mask &= mask - 1;
                    }
                    currentStrIdx += 16;
                }
                if (currentStrIdx > 0) {
                    prevByte = compressed[currentStrIdx - 1];
                } else {
                    prevByte = 0;
                }
                while (currentStrIdx <= maxIdx && (!transitionArrayq0[compressed[currentStrIdx]] || prevByte == 255)) {
                    prevByte = compressed[currentStrIdx];
                    ++currentStrIdx;
                }
                if (currentStrIdx > maxIdx)
                    return false;
                q0:
                *strIdx = currentStrIdx;
                switch(compressed[currentStrIdx]){
                    case 65:
                        q = FwdState::q4;
                        level = 2;
                        break;
                    case 76:
                        q = FwdState::q2;
                        level = 2;
                        break;
                }
                break;
            }
            case FwdState::q1:
                switch(compressed[*strIdx]){
                    case 255:
                        q = FwdState::q5;
                        level = 4;
                        break;
                    case 76:
                        q = FwdState::q2;
                        level = 2;
                        break;
                    case 8:
                        q = FwdState::E2;
                        level = 0;
                        break;
                    case 65:
                        q = FwdState::q4;
                        level = 2;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 3;
                        break;
                }
                break;
            case FwdState::q2:
                switch(compressed[*strIdx]){
                    case 59:
                        q = FwdState::q1;
                        level = 1;
                        break;
                    case 76:
                        q = FwdState::q2;
                        level = 2;
                        break;
                    case 255:
                        q = FwdState::q5;
                        level = 4;
                        break;
                    case 65:
                        q = FwdState::q4;
                        level = 2;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 3;
                        break;
                }
                break;
            case FwdState::q3:
                switch(compressed[*strIdx]){
                    case 76:
                        q = FwdState::q2;
                        level = 2;
                        break;
                    case 65:
                        q = FwdState::q4;
                        level = 2;
                        break;
                    case 168:
                    case 156:
                    case 118:
                    case 87:
                    case 41:
                    case 28:
                    case 11:
                    case 6:
                        q = FwdState::E1;
                        level = 0;
                        break;
                    case 255:
                        q = FwdState::q5;
                        level = 4;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 3;
                        break;
                }
                break;
            case FwdState::q4:
                switch(compressed[*strIdx]){
                    case 14:
                        q = FwdState::q3;
                        level = 1;
                        break;
                    case 76:
                        q = FwdState::q2;
                        level = 2;
                        break;
                    case 255:
                        q = FwdState::q5;
                        level = 4;
                        break;
                    case 65:
                        q = FwdState::q4;
                        level = 2;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 3;
                        break;
                }
                break;
            case FwdState::q5:
                switch(compressed[*strIdx]){
                    default:
                        q = FwdState::q0;
                        level = 3;
                        break;
                }
                break;
        }
        *strIdx += 1;
    }
    switch(q){
        case FwdState::E1:
            *symIdx = 1;
            *strIdx -= 1;
            return true;
        case FwdState::E2:
            *symIdx = 2;
            *strIdx -= 1;
            return true;
        default:
            return false;
    }
}

extern "C" bool parse_cpu_under(const uint8_t* compressed, size_t len) {
    if (len < 3)
        return false;
    int64_t strIdx = 0;
    uint8_t symIdx;
    return parseFwdState(compressed, len, &strIdx, &symIdx);
}