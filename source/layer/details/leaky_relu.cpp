#include "kuiper/layer/details/leaky_relu.hpp"
#include "kuiper/layer/abstract/layer_factory.hpp"
#include "kuiper/utils/math/fmath.hpp"

namespace kuiper_infer {

LeakyRelu::LeakyRelu(float negative_slope)
    : NonParamLayer("nn.LeakyRelu"), negative_slope_(negative_slope) {}

StatusCode LeakyRelu::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  /// LeakyRelu带有参数，无法放入simd中
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the softmax layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the softmax layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the softmax "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the LeakyRelu layer has an empty tensor " << i << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(output != nullptr && output->shapes() == input->shapes())
        << "The input and output tensor shapes of the LeakyRelu layer do not match " << i << " th";

    int64_t index;
    int64_t packet_size;
    int64_t in_size = static_cast<int64_t>(input->size());
    const float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
    packet_size = 8;
    __m256 zero = _mm256_setzero_ps();
    __m256 negative_slope = _mm256_set1_ps(negative_slope_);
    for (index = 0; index <= in_size - packet_size; index += packet_size) {
      __m256 p = _mm256_loadu_ps(in_ptr);
      /// 创建掩码，如果元素大于等于0，则对应位置全为1，否则为0
      __m256 mask = _mm256_cmp_ps(p, zero, _CMP_GE_OS);
      /// 使用掩码和逻辑运算符混合原始值和 negative_slope * x
      /// _mm256_blendv_ps(a, b, mask)
      /// if mask[i] : dst[i] = b[i]       # >= 0
      /// else  dst[i] = a[i]              # <  0
      __m256 result = _mm256_blendv_ps(_mm256_mul_ps(p, negative_slope), p, mask);
      /// 存储结果
      _mm256_storeu_ps(out_ptr, result);
      in_ptr += packet_size;
      out_ptr += packet_size;
    }
#ifdef __SSE2__
    packet_size = 4;
    __m128 zero128 = _mm_setzero_ps();
    __m128 negative_slope128 = _mm_set1_ps(negative_slope_);
    for (; index <= in_size - packet_size; index += packet_size) {
      __m128 p = _mm_loadu_ps(in_ptr);
      __m128 mask = _mm_cmp_ps(p, zero128, _CMP_GE_OS);
      __m128 result = _mm_blendv_ps(_mm_mul_ps(p, negative_slope128), p, mask);
      _mm_storeu_ps(out_ptr, result);
      in_ptr += packet_size;
      out_ptr += packet_size;
    }
#endif
#endif
    if (index < in_size) {
      while (index < in_size) {
        float value = input->index(index);
        if (value >= 0.f) {
          output->index(index) = value;
        } else {
          output->index(index) = value * negative_slope_;
        }
        index += 1;
      }
    }
  }

  return StatusCode::kSuccess;
}

StatusCode LeakyRelu::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& leakyrelu_layer) {
  if (!op) {
    LOG(ERROR) << "The leakyrelu operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the leakyrelu layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (params.find("negative_slope") == params.end()) {
    LOG(ERROR) << "Can't find the negative_slope parameter";
    return StatusCode::kParseParamError;
  }

  auto negative_slope_param =
      std::dynamic_pointer_cast<RuntimeParameterFloat>(params.at("negative_slope"));
  if (!negative_slope_param) {
    LOG(ERROR) << "Can't find the negative_slope parameter";
    return StatusCode::kParseParamError;
  }

  const float negative_slope = negative_slope_param->value;

  leakyrelu_layer = std::make_shared<LeakyRelu>(negative_slope);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kLeakyReluCreateInstance(LeakyRelu::CreateInstance, "nn.LeakyRelu");
}  // namespace kuiper_infer
