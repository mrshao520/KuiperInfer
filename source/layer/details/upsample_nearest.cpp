#include "kuiper/layer/details/upsample_nearest.hpp"
#include "kuiper/layer/abstract/layer_factory.hpp"
#include "kuiper/layer/details/upsample.hpp"

namespace kuiper_infer {
StatusCode UpsampleNearestLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                                std::shared_ptr<Layer<float>>& upsample_layer) {
  if (!op) {
    LOG(ERROR) << "The upsample_nearest operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the upsample_nearest layer is empty.";
  }

  if (params.find("scale_factor") == params.end()) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return StatusCode::kParseParamError;
  }

  auto scales = std::dynamic_pointer_cast<RuntimeParameterFloatArray>(params.at("scale_factor"));
  if (scales == nullptr) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return StatusCode::kParseParamError;
  }

  if (scales->value.size() != 2) {
    LOG(ERROR) << "Scale factor need two dimension";
    return StatusCode::kParseParamError;
  }

  const float scale_h = scales->value.at(0);
  const float scale_w = scales->value.at(1);
  // scale放大的倍数大于0
  if (scale_h <= 0 || scale_w <= 0) {
    LOG(ERROR) << "The parameter scale height and scale width should be greater than zero.";
    return StatusCode::kParseParamError;
  }

  upsample_layer =
      std::make_shared<UpSampleLayer>(scale_h, scale_w, UpSampleMode::kModeNearest, false);

  return StatusCode::kSuccess;
}
LayerRegistererWrapper kUpsamplerNearestCreateInstance(UpsampleNearestLayer::CreateInstance,
                                                       "F.upsample_nearest", "nn.upsample_nearest");
}  // namespace kuiper_infer
