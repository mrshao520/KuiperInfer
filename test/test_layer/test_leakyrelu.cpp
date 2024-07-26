#include <glog/logging.h>
#include <gtest/gtest.h>
#include "kuiper/data/tensor.hpp"
#include "kuiper/layer/details/leaky_relu.hpp"

TEST(test_layer, forward_leakyrelu1) {
  using namespace kuiper_infer;
  float negative_slope = 1e-2;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 32, 128);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  LeakyRelu relu_layer(negative_slope);
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    input_->Transform([negative_slope](const float f) {
      if (f >= 0) {
        return f;
      } else {
        return f * negative_slope;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_leakyrelu2) {
  using namespace kuiper_infer;
  float negative_slope = 1e-2;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 7);
  input->RandN();
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(1, 1, 7);
  input1->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input1);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(2);

  LeakyRelu relu_layer(negative_slope);
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    LOG(INFO) << "input: ";
    input_->Show();
    input_->Transform([negative_slope](const float f) {
      if (f >= 0) {
        return f;
      } else {
        return f * negative_slope;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    LOG(INFO) << "output: ";
    output_->Show();
    LOG(INFO) << "result: ";
    input_->Show();
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_leakyrelu3) {
  using namespace kuiper_infer;
  float negative_slope = 1e-2;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 15);
  input->RandN();
  LOG(INFO) << "input : ";
  input->Show();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  LeakyRelu relu_layer(negative_slope);
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    input_->Transform([negative_slope](const float f) {
      if (f >= 0) {
        return f;
      } else {
        return f * negative_slope;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    LOG(INFO) << "output : ";
    output_->Show();
    LOG(INFO) << "result : ";
    input_->Show();
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}