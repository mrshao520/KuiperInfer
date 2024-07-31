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

// Created by fss on 22-12-25.
#include "kuiper/layer/details/cat.hpp"
#include "kuiper/layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
CatLayer::CatLayer(int32_t dim) : NonParamLayer("cat"), dim_(dim) {}

StatusCode CatLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                             std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  StatusCode status_code = Check(inputs, outputs);
  if (status_code != StatusCode::kSuccess) {
    return status_code;
  }

  /// 输入张量维度 2 <= dims <= 4
  /// 0维是批量大小，无法拼接
  /// cat测试 3 和 4 失败原因
  /// 在KuiperInfer中，作者对维度进行缩减，如(1, 2, 3) -> (2, 3)，
  /// 使其难以获取原始维度，无法进行准确拼接
  int32_t dims = inputs[0]->dims() + 1;
  int positive_axis = dim_ < 0 ? dims + dim_ : dim_;
  CHECK(positive_axis < dims && positive_axis >= 1)
      << "Dimension out of range (expected to be in range of [" << -dims << " , " << dims
      << "], but got " << dim_ << ")";

  /// Tensor中使用cube，只支持 3d
  /// 故可以增加到4维，方便处理
  positive_axis += 4 - dims;

  const uint32_t input_size = inputs.size();
  const uint32_t output_size = outputs.size();

#pragma omp parallel for num_threads(output_size)
  for (uint32_t i = 0; i < output_size; ++i) {
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    uint32_t in_channels = inputs.at(i)->channels();
    uint32_t in_rows = inputs.at(i)->rows();
    uint32_t in_cols = inputs.at(i)->cols();
    uint32_t total_size = 0;

    if (positive_axis == 1) {  /// dim-axis: 4-1
      for (uint32_t j = i; j < input_size; j += output_size) {
        const auto& input = inputs.at(j);
        // CHECK(in_channels == input->channels())
        //     << "Sizes of tensors must match except in dimension " << positive_axis << "."
        //     << "Expected size " << in_channels << " but got size " << input->channels()
        //     << " for tensor number " << j << " in the list.";
        CHECK(in_rows == input->rows())
            << "Sizes of tensors must match except in dimension " << positive_axis << "."
            << "Expected size " << in_rows << " but got size " << input->rows()
            << " for tensor number " << j << " in the list.";
        CHECK(in_cols == input->cols())
            << "Sizes of tensors must match except in dimension " << positive_axis << "."
            << "Expected size " << in_cols << " but got size " << input->cols()
            << " for tensor number " << j << " in the list.";
        total_size += input->channels();
      }

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(total_size, in_rows, in_cols);
        outputs.at(i) = output;
      } else {
        CHECK(output->channels() == total_size) << "The cat layer get an incorrectly sized tensor."
                                                << "The tensor channel size should be "
                                                << total_size << ", but get " << output->channels();
        CHECK(output->rows() == in_rows)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor row size should be " << in_rows << ", but get " << output->rows();
        CHECK(output->cols() == in_cols)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor cols size should be " << in_cols << ", but get " << output->cols();
      }

      uint32_t offset = 0;
      for (uint32_t j = i; j < input_size; j += output_size) {
        const auto& input = inputs.at(j);
        memcpy(output->raw_ptr(offset), input->raw_ptr(), sizeof(float) * input->size());
        offset += input->size();
      }
    }

    if (positive_axis == 2) {  /// dim-axis: 4-2 3-1
      for (uint32_t j = i; j < input_size; j += output_size) {
        const auto& input = inputs.at(j);
        CHECK(in_channels == input->channels())
            << "Sizes of tensors must match except in dimension " << positive_axis << "."
            << "Expected size " << in_channels << " but got size " << input->channels()
            << " for tensor number " << j << " in the list.";
        // CHECK(in_rows == input->rows())
        //     << "Sizes of tensors must match except in dimension " << positive_axis << "."
        //     << "Expected size " << in_rows << " but got size " << input->rows()
        //     << " for tensor number " << j << " in the list.";
        CHECK(in_cols == input->cols())
            << "Sizes of tensors must match except in dimension " << positive_axis << "."
            << "Expected size " << in_cols << " but got size " << input->cols()
            << " for tensor number " << j << " in the list.";
        total_size += input->rows();
      }

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(in_channels, total_size, in_cols);
        outputs.at(i) = output;
      } else {
        CHECK(output->channels() == in_channels)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor channel size should be " << in_channels << ", but get "
            << output->channels();
        CHECK(output->rows() == total_size)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor row size should be " << total_size << ", but get " << output->rows();
        CHECK(output->cols() == in_cols)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor cols size should be " << in_cols << ", but get " << output->cols();
      }
#pragma omp parallel for num_threads(in_channels)
      for (uint32_t c = 0; c < in_channels; ++c) {
        /// 记录已拷贝的行数
        uint32_t row_offset = 0;
        for (uint32_t j = i; j < input_size; j += output_size) {
          const auto& input = inputs.at(j);
          for (uint32_t col = 0; col < in_cols; ++col) {
            /// 计算通道内每一列的起始地址
            uint32_t offset = c * output->plane_size() + col * output->rows() + row_offset;
            memcpy(output->raw_ptr(offset), input->slice(c).colptr(col),
                   sizeof(float) * input->rows());
          }
          row_offset += input->rows();
        }
      }
    }

    if (positive_axis == 3) {  /// dim-axis: 4-3 3-2 2-1
      for (uint32_t j = i; j < input_size; j += output_size) {
        const auto& input = inputs.at(j);
        CHECK(in_channels == input->channels())
            << "Sizes of tensors must match except in dimension " << positive_axis << "."
            << "Expected size " << in_channels << " but got size " << input->channels()
            << " for tensor number " << j << " in the list.";
        CHECK(in_rows == input->rows())
            << "Sizes of tensors must match except in dimension " << positive_axis << "."
            << "Expected size " << in_rows << " but got size " << input->rows()
            << " for tensor number " << j << " in the list.";
        // CHECK(in_cols == input->cols())
        //     << "Sizes of tensors must match except in dimension " << positive_axis << "."
        //     << "Expected size " << in_cols << " but got size " << input->cols()
        //     << " for tensor number " << j << " in the list.";
        total_size += input->cols();
      }

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(in_channels, in_rows, total_size);
        outputs.at(i) = output;
      } else {
        CHECK(output->channels() == in_channels)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor channel size should be " << in_channels << ", but get "
            << output->channels();
        CHECK(output->rows() == in_rows)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor row size should be " << in_rows << ", but get " << output->rows();
        CHECK(output->cols() == total_size)
            << "The cat layer get an incorrectly sized tensor."
            << "The tensor cols size should be " << total_size << ", but get " << output->cols();
      }
#pragma omp parallel for num_threads(in_channels)
      for (uint32_t c = 0; c < in_channels; ++c) {
        uint32_t offset = c * output->plane_size();
        for (uint32_t j = i; j < input_size; j += output_size) {
          const auto& input = inputs.at(j);
          /// armadillo 为列主序，直接拷贝即可
          memcpy(output->raw_ptr(offset), input->slice(c).memptr(),
                 sizeof(float) * input->plane_size());
          offset += input->plane_size();
        }
      }
    }
  }

  return StatusCode::kSuccess;
}

StatusCode CatLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                    std::shared_ptr<Layer<float>>& cat_layer) {
  if (!op) {
    LOG(ERROR) << "The cat operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the cat layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParamError;
  }

  auto dim_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParamError;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return StatusCode::kSuccess;
}

StatusCode CatLayer::Check(const std::vector<sftensor>& inputs,
                           const std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  for (const auto& input_data : inputs) {
    if (input_data == nullptr || inputs.empty()) {
      return StatusCode::kInferInputsEmpty;
    }
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the cat layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  const uint32_t input_size = inputs.size();
  const uint32_t output_size = outputs.size();
  if (input_size % output_size != 0) {
    LOG(ERROR) << "The input and output tensor array size of cat layer do not match"
               << "(input_size is " << input_size << " and output_size is " << output_size;
    return StatusCode::kInferDimMismatch;
  }

  return StatusCode::kSuccess;
}

LayerRegistererWrapper kCatCreateInstance(CatLayer::CreateInstance, "torch.cat");
}  // namespace kuiper_infer