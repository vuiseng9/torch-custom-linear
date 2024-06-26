#include <torch/extension.h>
#include <torch/nn/functional/linear.h>
#include <optional>

namespace F = torch::nn::functional;

// intent is to create an alternative kernel for linear layer by using torch C API
// just to demonstrate the capability of torch custom op extension
// we will not implement backward since we focus of inference
// we can also reuse parent implementation if we inherit nn.Linear

// implementation below is adapted from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/functional/linear.h

torch::Tensor dense_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {
  if (input.dim() == 2 && bias.has_value()) {
    // fused op is marginally faster
    return torch::addmm(bias.value(), input, weight.t());
  } else {
    auto output = input.matmul(weight.t());
    if (bias.has_value()) {
      output += bias.value();
    }
    return output;
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_linear", &dense_linear_forward, "dense linear forward",
  pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = nullptr);
}
