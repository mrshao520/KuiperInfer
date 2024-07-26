#ifndef KUIPER_INFER_SOURCE_LAYER_UPSAMPLE_BILINEAR_HPP_
#define KUIPER_INFER_SOURCE_LAYER_UPSAMPLE_BILINEAR_HPP_
#include "kuiper/layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {
class UpsampleBilinearLayer : public NonParamLayer {
 public:
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& upsample_layer);
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_UPSAMPLE_BILINEAR_HPP_