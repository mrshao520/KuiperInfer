#include <glog/logging.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../image_util.hpp"
#include "kuiper/data/tensor.hpp"
#include "kuiper/runtime/runtime_ir.hpp"
#include "kuiper/tick.hpp"

kuiper_infer::sftensor PreProcessImage(const cv::Mat& image, const int32_t input_h,
                                       const int32_t input_w) {
  assert(!image.empty());
  using namespace kuiper_infer;
  const int32_t input_c = 3;

  // int stride = 32;
  // cv::Mat out_image;
  // Letterbox(image, out_image, {input_h, input_w}, stride, {114, 114, 114}, true);

  cv::Mat rgb_image;
  cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat normalize_image;
  rgb_image.convertTo(normalize_image, CV_32FC3, 1. / 255.);

  std::vector<cv::Mat> split_images;
  cv::split(normalize_image, split_images);
  assert(split_images.size() == input_c);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_c, input_h, input_w);
  input->Fill(0.f);

  int index = 0;
  int offset = 0;
  uint32_t total = input_h * input_w;
  for (const auto& split_image : split_images) {
    assert(split_image.total() == total);
    const cv::Mat& split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data, sizeof(float) * split_image.total());
    index += 1;
    offset += split_image.total();
  }

  return input;
}

void PostProcessImage(const std::vector<kuiper_infer::sftensor>& outputs) {
  int32_t size = outputs.size();

  for (int32_t i = 0; i < size; ++i) {
    const auto& img = outputs.at(i);
    const int32_t input_h = img->rows();
    const int32_t input_w = img->cols();
    const int32_t input_c = img->channels();
    const int32_t total = input_h * input_w;
    cv::Mat cv_img(input_h, input_w, CV_32FC3);
    for (int32_t c = 0; c < input_c; ++c) {
      memcpy(cv_img.ptr<cv::Vec3f>(0, i), img->slice(c).memptr(), total * sizeof(float));
    }
    cv_img.convertTo(cv_img, CV_8UC3, 255.0);
    cv::imwrite(std::string("output/sr_") + std::to_string(i) + std::string(".jpg"), cv_img);
  }
  return;
}

void RealesrganDemo(const std::vector<std::string>& image_paths, const std::string& param_path,
                    const std::string& bin_path, const uint32_t batch_size) {
  using namespace kuiper_infer;

  LOG(INFO) << "Build Graph!";
  LOG(INFO) << "param path: " << param_path;
  LOG(INFO) << "bin   path: " << bin_path;
  RuntimeGraph graph(param_path, bin_path);
  graph.Build();
  CHECK(batch_size == image_paths.size()) << "patch size unmatch!";

  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto& input_image = cv::imread(image_paths.at(i));
    uint32_t input_h = input_image.rows;
    uint32_t input_w = input_image.cols;
    sftensor input = PreProcessImage(input_image, input_h, input_w);
    assert(input->rows() == input_h);
    assert(input->cols() == input_w);
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  graph.set_inputs("pnnx_input_0", inputs);
  for (int i = 0; i < 1; ++i) {
    graph.Forward(true);
  }
  outputs = graph.get_outputs("pnnx_output_0");
  assert(outputs.size() == inputs.size() && outputs.size() == batch_size);

  PostProcessImage(outputs);

  return;
}

int main() {
  const uint32_t batch_szie = 8;
  LOG(INFO) << "----- real_eargan -----";
  std::vector<std::string> image_paths;
  for (uint32_t i = 0; i < batch_szie; ++i) {
    const std::string& image_path = "./imgs/black_cat.jpg";
    image_paths.push_back(image_path);
  }

  const std::string& param_path = "tmp/real_esrgan/RealESRGAN_x4plus_anime_6B.pnnx.param";
  const std::string& bin_path = "tmp/real_esrgan/RealESRGAN_x4plus_anime_6B.pnnx.bin";

  RealesrganDemo(image_paths, param_path, bin_path, batch_szie);

  return 0;
}