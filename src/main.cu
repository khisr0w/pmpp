#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "types.h"
#include "utils.h"
#include "vec_add.cu"
#include "color_to_grayscale.cu"
#include "image_blur.cu"
#include "matmul.cu"

typedef void (*matmul_func)(fmatrix *, fmatrix *, fmatrix *);

internal bool
fcompare(f64 a, f64 b) {
    f64 threshold = 0.001;

    return fabs(a - b) <= threshold;
}

internal void
test_mat_vec() {
    f32 W_data[] = {
        1.9, 2.0, 3.4, 
        4.0, 5.3, 6.1, 
        7.1, 8.3, 9.7, 
        7.2, 8.0, 9.0,
    };
    fmatrix W = {W_data, 4, 3};

    f32 x_data[] = { 7.0, 8.0, 5.0, };
    fvec x = {x_data, 3};

    f32 result_data[] = { 0.0, 0.0, 0.0, 0.0 };
    fvec result = {result_data, 4};

    f32 answer_data[] = { 46.3, 100.9, 164.6, 159.4 };
    fvec answer = {answer_data, 4};

    mat_vec(&W, &x, &result);

    bool is_equal = true;
    for(u64 idx = 0; idx < result.n; ++idx) {
        f32 answer_value = answer.data[idx];
        f32 result_value = result.data[idx];
        is_equal = is_equal && fcompare(answer_value, result_value);
    }
    test("Wx == answer", is_equal, false);
}

internal void
test_matmul_matrix(matmul_func func) {
    f32 A_data[] = {
        1.0, 2.0, 3.0, 7.0,
        4.0, 5.0, 6.0, 8.0,
        7.0, 8.0, 9.0, 9.0,
    };
    fmatrix A = {A_data, 3, 4};

    f32 B_data[] = {
        7.0, 8.0,
        4.0, 5.0,
        1.0, 2.0,
        9.0, 6.0
    };
    fmatrix B = {B_data, 4, 2};

    f32 C_data[] = {
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
    };
    fmatrix C = {C_data, 3, 2};

    f32 answer_data[] = {
         81.0,  66.0, 
        126.0, 117.0, 
        171.0, 168.0, 
    };
    fmatrix answer = {answer_data, 3, 2};

    // matmul_element_per_thread(&A, &B, &C);
    func(&A, &B, &C);

    bool is_equal = true;
    for(u64 idx = 0; idx < C.n*C.m; ++idx) {
        f32 answer_value = answer.data[idx];
        f32 c_value = C.data[idx];
        is_equal = is_equal && fcompare(answer_value, c_value);
    }
    test("c == answer", is_equal, false);
    // print_matrix(C);

    f32 A2_data[] = {
        7.5626e-01, -6.2879e-01,  6.6402e-01, -1.8664e+00,  8.8203e-01, -1.1602e+00,  3.2769e-01,
        5.8758e-01, -6.6284e-02,  1.4003e+00, -1.1359e+00,  5.5899e-03, -7.2212e-01,  1.6636e-02,
        1.7378e-01,  9.2214e-01, -1.0156e+00, -4.1563e-01, -4.8972e-01, -7.7406e-01, -2.5812e-01,
       -7.4778e-01,  3.1948e-01, -1.3699e-01,  6.0236e-02,  1.0666e+00, 6.3561e-01,  4.4254e-01,
       -1.7772e+00, -3.8438e-01, -1.2958e+00, -2.3959e-01,  9.9197e-01, 1.8762e-01, -3.1529e-01,
       -4.1526e-01, -1.9794e+00, -1.3937e+00,  2.1311e+00, -8.9296e-01, 2.8673e-01, -3.4877e-01,
       -9.4351e-01, -1.9868e+00, -6.6522e-01, -1.5077e+00, -1.4260e+00, 1.0568e-03, -1.4656e-01,
       -2.2116e-01,  5.2986e-01, -1.0591e+00, -4.2071e-01, -1.2633e+00, -1.2439e+00,  1.5019e-01,
        4.7646e-01,  6.2171e-02, -1.4004e+00,  7.6000e-01,  1.1745e+00, -2.6465e-01,  9.6928e-02,
       -7.4139e-01,  4.4239e-01,  4.9145e-01,  7.5860e-02,  7.5682e-02, -1.5973e+00, -2.8502e-01,
        6.1407e-01, -9.7959e-01,  1.0673e-01, -1.1159e+00, -3.2390e-01, 5.6488e-01, -9.3294e-01,
       -2.9656e-01, -5.9523e-01,  1.3524e+00, -5.0744e-01, -6.0088e-01, -8.3856e-01, -5.4524e-01
    };
    fmatrix A2 = {A2_data, 12, 7};

    f32 B2_data[] = {
        -8.3474e-01, -5.1291e-02,  2.7527e-01, -1.2156e+00,  1.4691e+00, 2.2403e-01, -6.0687e-01,  1.7320e-02, -1.7047e-01, -5.0509e-01, 1.0304e+00,
         2.0067e+00,  8.9837e-01,  8.0027e-01,  1.5478e+00,  1.6955e-01, 1.3303e-01, -1.0505e+00, -1.3562e+00, -5.1863e-01, -1.2190e+00, -6.7796e-01,
        -1.6147e+00, -2.7281e-01,  2.2760e-01,  7.9381e-01,  1.0071e+00, -1.5595e+00,  1.6915e-01, -1.7497e-01, -1.4065e+00,  1.5556e+00, -1.1173e+00,
        -3.1720e-01,  7.2996e-01,  3.3483e-01,  4.9490e-01,  6.6934e-01, -4.6770e-01,  1.0749e-03, -1.7878e+00, -7.1244e-01, -2.9539e-01, -5.2781e-01,
         9.9910e-01,  2.4286e+00, -1.8033e+00, -1.0388e+00, -6.3191e-02, 2.6968e-01,  2.0522e+00,  2.0634e+00,  1.9620e+00, -3.4411e-01, 1.9162e-01,
         1.3974e+00,  2.0463e+00, -2.7449e-01,  1.1612e+00,  7.3464e-01, 1.8702e-01,  5.9305e-02, -5.6213e-01, -8.8042e-01,  5.0839e-01, -1.3970e+00,
         2.5368e+00, -1.2295e+00, -4.7158e-01, -2.4461e-01,  6.9772e-01, 1.2937e-01, -8.7902e-01, -1.1619e+00, -5.7300e-01,  5.8945e-01, 4.5929e-01
    };
    fmatrix B2 = {B2_data, 7, 11};

    f32 C2_data[12*11];
    memset(C2_data, sizeof(f32), 0);
    fmatrix C2 = {C2_data, 12, 11};

    f32 answer2_data[] = {
        -2.2820, -2.7822, -2.1955, -4.6329, -0.2556, -0.0135,  1.7651,  6.1779, 3.1572,  1.2686,  3.3891,
        -3.4855, -2.7855,  0.2274, -1.1159,  0.9826, -1.6610, -0.0973,  2.2840, -0.5888,  1.9386,  0.7029,
         1.2514, -1.6628,  1.6328, -0.1228, -1.6072,  1.6296, -2.0704, -0.6023, 1.0853, -3.0461,  1.7769,
         4.5439,  3.7536, -2.2677,  0.8464, -0.4337,  0.5242,  1.9325,  0.7993, 1.3910, -0.0257, -1.3462,
         3.3338,  3.1052, -2.8636, -0.3173, -4.2862,  1.9454,  3.5868,  3.4533, 4.4573, -1.0106, -0.2132,
        -3.4273, -0.9742,  0.3940, -1.2647, -0.8991,  0.5881,  0.5890, -2.4873, -0.2651,  0.0725,  0.6146,
        -3.4421, -5.9364,  0.1343, -1.6841, -3.4135,  0.8635, -0.2518,  2.7174, 0.4864,  2.7139,  1.5718,
         0.4721, -5.3290,  2.5299, -0.1288, -2.3125,  1.3154, -3.4003, -1.8669, 0.0828, -2.1665,  2.3828,
         2.7966,  3.1598, -1.9743, -2.7696, -0.3921,  2.2232,  1.7188,  1.2697, 3.7965, -3.2009,  2.2515,
        -2.1905, -2.3776,  0.7236,  0.1499, -1.8455, -1.2243,  0.3795,  0.5508, 0.8697, -0.4288,  0.4620,
        -4.1978, -0.2388, -0.0952, -1.5096, -0.1188,  0.2603,  0.8621,  3.4136, 0.4500,  1.2284,  0.4869,
        -6.1250, -3.7638,  1.1508,  0.0454, -0.4728, -2.4068,  0.2299,  1.3377, -1.3096,  2.5881, -0.3393
    };
    fmatrix answer2 = {answer2_data, 12, 11};

    func(&A2, &B2, &C2);

    is_equal = true;
    for(u64 idx = 0; idx < C2.n*C2.m; ++idx) {
        f32 answer_value = answer2.data[idx];
        f32 c_value = C2.data[idx];
        is_equal = is_equal && fcompare(answer_value, c_value);
    }
    test("c2 == answer2", is_equal, false);
    // print_matrix(C2);
}

i32 main(i32 argc, char *argv[]) {
#if 0
    color_to_grayscale_conversion("assets/rgb_bird.jpg");
    image_blur("assets/rgb_bird.jpg", 12);
    matmul_element_per_thread((f32 *)a, (f32 *)b, (f32 *)c, n);

    printf("Per Element\n"); test_matmul_matrix(matmul_element_per_thread); printf("\n");
    printf("Per Row\n"); test_matmul_matrix(matmul_row_per_thread); printf("\n");
    printf("Per Column\n"); test_matmul_matrix(matmul_col_per_thread); printf("\n");
    test_mat_vec();
#endif

    return 0;
}
