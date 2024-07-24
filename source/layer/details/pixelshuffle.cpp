#include "kuiper/layer/details/pixelshuffle.hpp"
#include "kuiper/data/tensor.hpp"
#include "kuiper/layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
PixelShuffleLayer::PixelShuffleLayer(int32_t upscale_factor)
    : NonParamLayer("PixelShuffle"), upscale_factor_(upscale_factor) {}

StatusCode PixelShuffleLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the pixelshuffle layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the pixelshuffle layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The output tensor array in the pixelshuffle layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  uint32_t batch = inputs.size();

#pragma omp parallel for if (batch > 1) num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    std::shared_ptr<Tensor<float>> input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the pixelshuffle layer has an empty tensor " << i << " th";
    const uint32_t input_channels = input->channels();
    const uint32_t input_rows = input->rows();
    const uint32_t input_cols = input->cols();
    uint32_t upscale_factor_pow = upscale_factor_ * upscale_factor_;
    CHECK(input_channels % upscale_factor_pow == 0)
        << "pixel_shuffle expects its input's 'channel' dimension to be divisible by "
        << "the square of upscale_factor, but " << input_channels << "is not divisible by "
        << upscale_factor_pow;
    uint32_t out_channels = input_channels / upscale_factor_pow;
    uint32_t out_rows = input_rows * upscale_factor_;
    uint32_t out_cols = input_cols * upscale_factor_;
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(out_channels, out_rows, out_cols);
      outputs.at(i) = output;
    } else {
      CHECK(out_channels == output->channels() && out_rows == output->rows() &&
            out_cols == output->cols())
          << "the output tensor array in pixel_shuffle has an incorrectly sized tensor " << i
          << " th";
    }

#pragma omp parallel for
    for (uint32_t output_c = 0; output_c < out_channels; ++output_c) {
      /// 循环遍历输出的通道
      arma::fmat& output_channel = output->slice(output_c);
      /// 每个输出通道对应 r^2 个输入通道
      for (int32_t sh = 0; sh < upscale_factor_; ++sh) {
        for (int32_t sw = 0; sw < upscale_factor_; ++sw) {
          /// 计算对应的输入通道号
          uint32_t input_c = output_c * upscale_factor_pow + sh * upscale_factor_ + sw;
          /// 获取输入通道指针，以列主序存储
          float* input_channel = input->slice(input_c).memptr();
          /// arma 列主序存储
          for (uint32_t i = 0; i < input_cols; ++i) {
            /// 获取输出指针
            float* output_ptr = output_channel.colptr(i * upscale_factor_ + sw) + sh;
            for (uint32_t j = 0; j < input_rows; ++j) {
              output_ptr[0] = input_channel[0];

              input_channel++;
              output_ptr += upscale_factor_;
            }
          }
        }
      }
    }
  }

  return StatusCode::kSuccess;
}

StatusCode PixelShuffleLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                             std::shared_ptr<Layer<float>>& pixelshuffle_layer) {
  if (!op) {
    LOG(ERROR) << "The pixel shuffle operator parameter in the layer is null pointer";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the pixelshuffle layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (params.find("upscale_factor") == params.end()) {
    LOG(ERROR) << "Can't find the upscale_factor parameter";
    return StatusCode::kParseParamError;
  }

  auto upscale_factor_param =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("upscale_factor"));
  if (!upscale_factor_param) {
    LOG(ERROR) << "Can't find the upscale_factor parameter";
    return StatusCode::kParseParamError;
  }

  const int32_t upscale_factor = upscale_factor_param->value;
  pixelshuffle_layer = std::make_shared<PixelShuffleLayer>(upscale_factor);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kPixelShuffleCreateInstance(PixelShuffleLayer::CreateInstance,
                                                   "torch.PixelShuffle");
}  // namespace kuiper_infer
