#include "../simple_mlp.h"

#include <cstdio>

int main() {
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[2];

    simple_mlp_infer(input, output);

    std::printf("Output: %f %f\n", output[0], output[1]);

    int predicted = (output[1] > output[0]) ? 1 : 0;
    std::printf("Predicted class: %d\n", predicted);

    return 0;
}
