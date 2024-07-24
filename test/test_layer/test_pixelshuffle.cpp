#include <glog/logging.h>
#include <gtest/gtest.h>
#include "kuiper/data/tensor.hpp"
#include "kuiper/layer/details/pixelshuffle.hpp"

TEST(test_layer, forward_pixelshuffle_size) {
  using namespace kuiper_infer;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_c = 4;
  const uint32_t input_h = 2;
  const uint32_t input_w = 3;
  const uint32_t input_size = 1;
  const uint32_t total = input_c * input_h * input_w;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    std::vector<float> values(total);
    for (uint32_t j = 0; j < total; ++j) {
      values.at(j) = float(j + input_size);
    }
    input->Fill(values);
    inputs.push_back(input);
  }
  for (auto& input : inputs) input->Show();
  auto input_raw_ptr = inputs.at(0)->raw_ptr();
  LOG(INFO) << input_raw_ptr[0];
  LOG(INFO) << input_raw_ptr[1];
  LOG(INFO) << input_raw_ptr[2];
}

TEST(test_layer, forward_pixelshuffle_upscale) {
  using namespace kuiper_infer;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const int32_t upscale = 2;
  const uint32_t input_c = 4;
  const uint32_t input_h = 2;
  const uint32_t input_w = 3;
  const uint32_t input_size = 1;
  const uint32_t total = input_c * input_h * input_w;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    std::vector<float> values(total);
    for (uint32_t j = 0; j < total; ++j) {
      values.at(j) = float(j + input_size);
    }
    input->Fill(values);
    inputs.push_back(input);
  }
  for (auto& input : inputs) input->Show();
  std::vector<std::shared_ptr<Tensor<float>>> outputs(input_size);
  PixelShuffleLayer layer(upscale);
  layer.Forward(inputs, outputs);
  for (auto& output : outputs) output->Show();
}

TEST(test_layer, forward_pixelshuffle_upscale3) {
  using namespace kuiper_infer;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const int32_t upscale = 3;
  const uint32_t input_c = 18;
  const uint32_t input_h = 3;
  const uint32_t input_w = 4;
  const uint32_t input_size = 2;
  const uint32_t total = input_c * input_h * input_w;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    std::vector<float> values(total);
    for (uint32_t j = 0; j < total; ++j) {
      values.at(j) = float(j + input_size);
    }
    input->Fill(values);
    inputs.push_back(input);
  }
  for (auto& input : inputs) input->Show();
  std::vector<std::shared_ptr<Tensor<float>>> outputs(input_size);
  PixelShuffleLayer layer(upscale);
  layer.Forward(inputs, outputs);
  for (auto& output : outputs) output->Show();
}