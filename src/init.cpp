#include <cstdint>
#include <cstring>

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
#define MAX_STATES 10 

extern "C" void upload_transition_matrix(const uint8_t* host_matrix);

extern "C" void init_pattern_something() {
    uint8_t host_matrix[MAX_STATES * 256];
    std::memset(host_matrix, Q0, sizeof(host_matrix));

    auto set_transitions = [&](int state, uint8_t byte, int next_state) {
        host_matrix[(state << 8) | byte] = next_state;
    };

    uint8_t prefixes[] = {21, 17, 33, 118, 67, 132};
    for (int s = 0; s < 9; ++s) {
        if (s == Q8) continue;
        set_transitions(s, 51, Q5);
        for (uint8_t p : prefixes) set_transitions(s, p, Q4);
        set_transitions(s, 255, Q8); 
    }

    set_transitions(Q1, 89, SOMETHING_ACCEPT);
    set_transitions(Q2, 79, Q1);
    set_transitions(Q3, 75, Q2);
    set_transitions(Q4, 167, Q3);
    set_transitions(Q5, 75, Q6);
    set_transitions(Q6, 79, Q7);
    set_transitions(Q7, 89, SOMETHING_ACCEPT);

    upload_transition_matrix(host_matrix);
}

extern "C" void init_pattern_under() {
    uint8_t host_matrix[MAX_STATES * 256];
    std::memset(host_matrix, Q0, sizeof(host_matrix));

    auto set_transitions = [&](int state, uint8_t byte, int next_state) {
        host_matrix[(state << 8) | byte] = next_state;
    };

    for (int s = Q0; s <= Q4; ++s) {
        set_transitions(s, 65, Q4);
        set_transitions(s, 76, Q2);
        set_transitions(s, 255, Q5);
    }

    set_transitions(Q1, 8, UNDER_E2);
    set_transitions(Q2, 59, Q1);
    set_transitions(Q4, 14, Q3);

    uint8_t q3_accepts[] = {168, 156, 118, 87, 41, 28, 11, 6};
    for (uint8_t c : q3_accepts) set_transitions(Q3, c, UNDER_E1);

    upload_transition_matrix(host_matrix);
}