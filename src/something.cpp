#include <cstddef>
#include <cstdint>
#include <bit>
#include <string_view>
#include <nmmintrin.h>
#include <pmmintrin.h>
constexpr bool transitionArrayq0[256] = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
constexpr uint8_t byteSetq0[16] = {132, 67, 118, 33, 17, 21, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__m128i byteVectorq0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byteSetq0));

enum class FwdState {
    E3,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    q8,
};

static bool parseFwdState(const uint8_t* compressed, size_t len, int64_t* strIdx, uint8_t* symIdx){
    *strIdx = 0;
    auto q = FwdState::q0;
    uint64_t level = 4;
    while (*strIdx + level <= len) {
        switch(q) {
            case FwdState::E3:
                *symIdx = 3;
                *strIdx -= 1;
                return true;
            case FwdState::q0:
            {
                int64_t maxIdx = len - level;
                uint8_t prevByte;
                int64_t currentStrIdx = *strIdx;
                while (currentStrIdx + 15 <= maxIdx) {
                    __m128i compressedVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compressed + currentStrIdx));
                    __m128i mask128i = _mm_cmpestrm(byteVectorq0, 7, compressedVec, 16, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
                    uint16_t mask = _mm_extract_epi16(mask128i, 0);
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
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    case 132:
                    case 67:
                    case 118:
                    case 33:
                    case 17:
                    case 21:
                        q = FwdState::q4;
                        level = 4;
                        break;
                }
                break;
            }
            case FwdState::q1:
                switch(compressed[*strIdx]){
                    case 21:
                    case 17:
                    case 33:
                    case 118:
                    case 67:
                    case 132:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 89:
                        q = FwdState::E3;
                        level = 0;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q2:
                switch(compressed[*strIdx]){
                    case 79:
                        q = FwdState::q1;
                        level = 1;
                        break;
                    case 21:
                    case 17:
                    case 33:
                    case 67:
                    case 132:
                    case 118:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q3:
                switch(compressed[*strIdx]){
                    case 21:
                    case 17:
                    case 33:
                    case 118:
                    case 67:
                    case 132:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 75:
                        q = FwdState::q2;
                        level = 2;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q4:
                switch(compressed[*strIdx]){
                    case 167:
                        q = FwdState::q3;
                        level = 3;
                        break;
                    case 21:
                    case 17:
                    case 33:
                    case 118:
                    case 67:
                    case 132:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q5:
                switch(compressed[*strIdx]){
                    case 75:
                        q = FwdState::q6;
                        level = 2;
                        break;
                    case 21:
                    case 17:
                    case 33:
                    case 118:
                    case 67:
                    case 132:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q6:
                switch(compressed[*strIdx]){
                    case 79:
                        q = FwdState::q7;
                        level = 1;
                        break;
                    case 21:
                    case 17:
                    case 33:
                    case 67:
                    case 132:
                    case 118:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q7:
                switch(compressed[*strIdx]){
                    case 21:
                    case 17:
                    case 33:
                    case 118:
                    case 67:
                    case 132:
                        q = FwdState::q4;
                        level = 4;
                        break;
                    case 255:
                        q = FwdState::q8;
                        level = 5;
                        break;
                    case 89:
                        q = FwdState::E3;
                        level = 0;
                        break;
                    case 51:
                        q = FwdState::q5;
                        level = 3;
                        break;
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
            case FwdState::q8:
                switch(compressed[*strIdx]){
                    default:
                        q = FwdState::q0;
                        level = 4;
                        break;
                }
                break;
        }
        *strIdx += 1;
    }
    switch(q){
        case FwdState::E3:
            *symIdx = 3;
            *strIdx -= 1;
            return true;
        default:
            return false;
    }
}

extern "C" bool parse_cpu_something(const uint8_t* compressed, size_t len) {
    if (len < 4)
        return false;
    int64_t strIdx = 0;
    uint8_t symIdx;
    return parseFwdState(compressed, len, &strIdx, &symIdx);
}