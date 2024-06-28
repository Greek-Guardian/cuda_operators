#include <torch/extension.h>
#include "abs.h"

void LZD_launch_abs(torch::Tensor &a,
                       int64_t n) {
    launch_abs((float *)a.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("abs",
          &LZD_launch_abs,
          "abs kernel warpper");
}