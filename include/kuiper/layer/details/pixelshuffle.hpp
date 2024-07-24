#ifndef KUIPER_INFER_SOURCE_LAYER_PIXELSHUFFLER_
#define KUIPER_INFER_SOURCE_LAYER_PIXELSHUFFLER_
#include "kuiper/layer/abstract/non_param_layer.hpp"
#include "kuiper/runtime/runtime_op.hpp"

namespace kuiper_infer {
class PixelShuffleLayer : public NonParamLayer {
 public:
  explicit PixelShuffleLayer(int32_t upscale_factor);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& pixelshuffle_layer);

 private:
  int32_t upscale_factor_;  ///< 提高空间分辨率的因子
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_PIXELSHUFFLER_