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

// Created by fss on 22-12-26.
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "kuiper/data/tensor.hpp"
#include "kuiper/layer/details/cat.hpp"

TEST(test_layer, cat1) {
  using namespace kuiper_infer;
  int input_size = 4;
  int output_size = 2;
  int input_channels = 6;
  int output_channels = 12;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (int i = 0; i < input_size / 2; ++i) {
    for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(i)->slice(input_channel),
                                     outputs.at(i)->slice(input_channel), "absdiff", 0.01f));
    }
  }

  for (int i = input_size / 2; i < input_size; ++i) {
    for (int input_channel = input_channels; input_channel < input_channels * 2; ++input_channel) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(i)->slice(input_channel - input_channels),
                                     outputs.at(i - output_size)->slice(input_channel), "absdiff",
                                     0.01f));
    }
  }
}

TEST(test_layer, cat2) {
  using namespace kuiper_infer;
  int input_size = 8;
  int output_size = 4;
  int input_channels = 64;
  int output_channels = 128;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (int i = 0; i < input_size / 2; ++i) {
    for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(i)->slice(input_channel),
                                     outputs.at(i)->slice(input_channel), "absdiff", 0.01f));
    }
  }

  for (int i = input_size / 2; i < input_size; ++i) {
    for (int input_channel = input_channels; input_channel < input_channels * 2; ++input_channel) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(i)->slice(input_channel - input_channels),
                                     outputs.at(i - output_size)->slice(input_channel), "absdiff",
                                     0.01f));
    }
  }
}

TEST(test_layer, cat3) {
  using namespace kuiper_infer;
  int input_size = 3;
  int input_channels = 1;

  int output_size = 1;
  int output_channels = 3;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 4, 4);
    input->Fill(float(i) + 1.f);
    inputs.push_back(input);
    input->Show();
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (int i = 0; i < input_size; ++i) {
    const arma::fmat& in_channel = inputs.at(i)->slice(0);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(in_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_layer, cat4) {
  using namespace kuiper_infer;
  int input_size = 6;
  int input_channels = 1;

  int output_size = 1;
  int output_channels = 6;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 4, 4);
    input->Fill(float(i) + 1.f);
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(0);
  const auto status = cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }

  for (int i = 0; i < input_size; ++i) {
    const arma::fmat& in_channel = inputs.at(i)->slice(0);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(in_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_layer, cat_new_2_1) {
  using namespace kuiper_infer;
  arma::fmat f1 = "0, 1, 2;";
  arma::fmat f2 = "3, 4;";
  arma::fmat f3 = "0, 1, 2, 3, 4;";

  sftensor data1 = std::make_shared<Tensor<float>>(3);
  sftensor data2 = std::make_shared<Tensor<float>>(2);
  sftensor result = std::make_shared<Tensor<float>>(5);
  data1->slice(0) = f1;
  data2->slice(0) = f2;
  result->slice(0) = f3;

  data1->Show();
  data2->Show();

  std::vector<sftensor> inputs;
  inputs.push_back(data1);
  inputs.push_back(data2);
  std::vector<sftensor> outputs(1);
  CatLayer cat_layer(1);
  cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->channels(), 1);
  ASSERT_EQ(outputs.at(0)->rows(), 1);
  ASSERT_EQ(outputs.at(0)->cols(), 5);
  outputs.at(0)->Show();

  for (uint32_t i = 0; i < outputs.at(0)->channels(); ++i) {
    const arma::fmat& res_channel = result->slice(i);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(res_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_layer, cat_new_3_1) {
  using namespace kuiper_infer;
  arma::fmat f1 =
      "0, 1, 2;"
      "4, 5, 6;";
  arma::fmat f2 = "-1, -2, -3;";
  arma::fmat f3 =
      "0, 1, 2;"
      "4, 5, 6;"
      "-1, -2, -3;";

  sftensor data1 = std::make_shared<Tensor<float>>(2, 3);
  sftensor data2 = std::make_shared<Tensor<float>>(1, 3);
  sftensor result = std::make_shared<Tensor<float>>(3, 3);
  data1->slice(0) = f1;
  data2->slice(0) = f2;
  result->slice(0) = f3;

  data1->Show();
  data2->Show();

  std::vector<sftensor> inputs;
  inputs.push_back(data1);
  inputs.push_back(data2);
  std::vector<sftensor> outputs(1);
  CatLayer cat_layer(1);
  cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->channels(), 1);
  ASSERT_EQ(outputs.at(0)->rows(), 3);
  ASSERT_EQ(outputs.at(0)->cols(), 3);
  outputs.at(0)->Show();

  for (uint32_t i = 0; i < outputs.at(0)->channels(); ++i) {
    const arma::fmat& res_channel = result->slice(i);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(res_channel, out_channel, "absdiff", 0.01f));
  }
}
TEST(test_layer, cat_new_3_2) {
  using namespace kuiper_infer;
  arma::fmat f1 =
      "0, 1, 2;"
      "4, 5, 6;";
  arma::fmat f2 =
      "-1, -2;"
      "-3, -4";
  arma::fmat f3 =
      "0, 1, 2, -1, -2;"
      "4, 5, 6, -3, -4;";

  sftensor data1 = std::make_shared<Tensor<float>>(2, 3);
  sftensor data2 = std::make_shared<Tensor<float>>(2, 2);
  sftensor result = std::make_shared<Tensor<float>>(2, 5);
  data1->slice(0) = f1;
  data2->slice(0) = f2;
  result->slice(0) = f3;

  data1->Show();
  data2->Show();

  std::vector<sftensor> inputs;
  inputs.push_back(data1);
  inputs.push_back(data2);
  std::vector<sftensor> outputs(1);
  CatLayer cat_layer(2);
  cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->channels(), 1);
  ASSERT_EQ(outputs.at(0)->rows(), 2);
  ASSERT_EQ(outputs.at(0)->cols(), 5);
  outputs.at(0)->Show();

  for (uint32_t i = 0; i < outputs.at(0)->channels(); ++i) {
    const arma::fmat& res_channel = result->slice(i);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(res_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_layer, cat_new_4_1) {
  using namespace kuiper_infer;
  arma::fmat f1 =
      "0, 1, 2;"
      "4, 5, 6;";

  arma::fmat f2 =
      "-1, -2, -3;"
      "-4, -5, -6;";

  sftensor data1 = std::make_shared<Tensor<float>>(2, 2, 3);
  sftensor data2 = std::make_shared<Tensor<float>>(1, 2, 3);
  sftensor result = std::make_shared<Tensor<float>>(3, 2, 3);

  data1->slice(0) = f1;
  data1->slice(1) = f1;
  data2->slice(0) = f2;

  result->slice(0) = f1;
  result->slice(1) = f1;
  result->slice(2) = f2;

  data1->Show();
  data2->Show();

  std::vector<sftensor> inputs;
  inputs.push_back(data1);
  inputs.push_back(data2);
  std::vector<sftensor> outputs(1);
  CatLayer cat_layer(1);
  cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->channels(), 3);
  ASSERT_EQ(outputs.at(0)->rows(), 2);
  ASSERT_EQ(outputs.at(0)->cols(), 3);
  outputs.at(0)->Show();

  for (uint32_t i = 0; i < outputs.at(0)->channels(); ++i) {
    const arma::fmat& res_channel = result->slice(i);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(res_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_layer, cat_new_4_2) {
  using namespace kuiper_infer;
  arma::fmat f1 =
      "0, 1, 2;"
      "4, 5, 6;"
      "7, 8, 9;";
  arma::fmat f2 =
      "-1, -2, -3;"
      "-4, -5, -6;";
  arma::fmat f3 =
      "0, 1, 2;"
      "4, 5, 6;"
      "7, 8, 9;"
      "-1, -2, -3;"
      "-4, -5, -6;";

  sftensor data1 = std::make_shared<Tensor<float>>(2, 3, 3);
  sftensor data2 = std::make_shared<Tensor<float>>(2, 2, 3);
  sftensor result = std::make_shared<Tensor<float>>(2, 5, 3);

  data1->slice(0) = f1;
  data1->slice(1) = f1;
  data2->slice(0) = f2;
  data2->slice(1) = f2;

  result->slice(0) = f3;
  result->slice(1) = f3;

  data1->Show();
  data2->Show();

  std::vector<sftensor> inputs;
  inputs.push_back(data1);
  inputs.push_back(data2);
  std::vector<sftensor> outputs(1);
  CatLayer cat_layer(2);
  cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->channels(), 2);
  ASSERT_EQ(outputs.at(0)->rows(), 5);
  ASSERT_EQ(outputs.at(0)->cols(), 3);
  outputs.at(0)->Show();

  for (uint32_t i = 0; i < outputs.at(0)->channels(); ++i) {
    const arma::fmat& res_channel = result->slice(i);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(res_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_layer, cat_new_4_3) {
  using namespace kuiper_infer;
  arma::fmat f1 =
      "0, 1, 2;"
      "4, 5, 6;";
  arma::fmat f2 =
      "-1, -2;"
      "-4, -5;";
  arma::fmat f3 =
      "0, 1, 2, -1, -2;"
      "4, 5, 6, -4, -5;";

  sftensor data1 = std::make_shared<Tensor<float>>(2, 2, 3);
  sftensor data2 = std::make_shared<Tensor<float>>(2, 2, 2);
  sftensor result = std::make_shared<Tensor<float>>(2, 2, 5);

  data1->slice(0) = f1;
  data1->slice(1) = f1;
  data2->slice(0) = f2;
  data2->slice(1) = f2;

  result->slice(0) = f3;
  result->slice(1) = f3;

  data1->Show();
  data2->Show();

  std::vector<sftensor> inputs;
  inputs.push_back(data1);
  inputs.push_back(data2);
  std::vector<sftensor> outputs(1);
  CatLayer cat_layer(3);
  cat_layer.Forward(inputs, outputs);
  ASSERT_EQ(outputs.at(0)->channels(), 2);
  ASSERT_EQ(outputs.at(0)->rows(), 2);
  ASSERT_EQ(outputs.at(0)->cols(), 5);
  outputs.at(0)->Show();

  for (uint32_t i = 0; i < outputs.at(0)->channels(); ++i) {
    const arma::fmat& res_channel = result->slice(i);
    const arma::fmat& out_channel = outputs.at(0)->slice(i);
    ASSERT_TRUE(arma::approx_equal(res_channel, out_channel, "absdiff", 0.01f));
  }
}