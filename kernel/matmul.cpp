#include <torch/extension.h>
#include "matmul.h"

void LZD_launch_matmul(torch::Tensor &C,
                       const torch::Tensor &A,
                       const torch::Tensor &B,
                       int64_t N,
                       int64_t K,
                       int64_t M) {
    launch_matmul((float *)C.data_ptr(),
                (const float *)A.data_ptr(),
                (const float *)B.data_ptr(),
                N,
                K,
                M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul",
          &LZD_launch_matmul,
          "matmul kernel warpper");
}