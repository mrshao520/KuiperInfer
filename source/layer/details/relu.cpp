// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-18.
#include "kuiper/layer/details/relu.hpp"
#include "kuiper/layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
using namespace activation;
ReluLayer::ReluLayer() : ActivationLayer(ActivationType::kActivationRelu, "nn.ReLU") {}

StatusCode ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode ReluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& relu_layer) {
  if (!op) {
    LOG(ERROR) << "The relu operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  relu_layer = std::make_shared<ReluLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kReluCreateInstance(ReluLayer::CreateInstance, "nn.ReLU");
}  // namespace kuiper_infer
