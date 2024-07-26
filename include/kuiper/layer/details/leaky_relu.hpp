#ifndef KUIPER_INFER_SOURCE_LAYER_LEAKY_RELU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LEAKY_RELU_HPP_
#include "kuiper/layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class LeakyRelu : public NonParamLayer {
 public:
  explicit LeakyRelu(float negative_slope = 1e-2);
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& leakyrelu_layer);

 private:
  float negative_slope_ = 1e-2; ///< 控制负斜率的角度，default: 1e-2
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_LEAKY_RELU_HPP_
