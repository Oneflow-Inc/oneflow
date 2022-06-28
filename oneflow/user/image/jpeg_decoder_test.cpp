/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <gtest/gtest.h>
#include <cstddef>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <opencv2/opencv.hpp>
#include "oneflow/user/image/jpeg_decoder.h"
#include "oneflow/user/image/image_util.h"

namespace oneflow {

// generate image
void GenerateImage(std::vector<uint8_t>& jpg, int w, int h) {
  std::vector<uint8_t> raw_data(w * h * 3);

  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
      uint8_t r = 0, g = 0, b = 0;
      if (i < w / 2 && j < h / 2) {
        r = 255;
        g = 0;
        b = 0;
      } else if ((i >= w / 2 && j < h / 2) || (i < w / 2 && j >= h / 2)) {
        r = 0;
        g = 255;
        b = 0;
      } else if ((i >= w / 2) && (j >= h / 2)) {
        r = 0;
        g = 0;
        b = 255;
      }

      raw_data[3 * (i * w + j)] = b;
      raw_data[3 * (i * w + j) + 1] = g;
      raw_data[3 * (i * w + j) + 2] = r;
    }
  }

  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
  compression_params.push_back(100);

  cv::Mat raw(h, w, CV_8UC3, (void*)raw_data.data(), cv::Mat::AUTO_STEP);
  cv::imencode(".jpg", raw, jpg);
}

TEST(JPEG, decoder) {
  constexpr size_t test_num = 3;
  std::vector<unsigned char> jpg;
  GenerateImage(jpg, 192, 192);
  std::seed_seq seq{1, 2, 3};
  std::vector<int64_t> seeds(test_num);
  seq.generate(seeds.begin(), seeds.end());

  for (int i = 0; i < test_num; i++) {
    cv::Mat libjpeg_image_mat;

    RandomCropGenerator libjpeg_random_crop_gen({0.1, 0.9}, {0.4, 0.6}, seeds[i], 1);
    RandomCropGenerator opencv_random_crop_gen({0.1, 0.9}, {0.4, 0.6}, seeds[i], 1);
    auto status = JpegPartialDecodeRandomCropImage(jpg.data(), jpg.size(), &libjpeg_random_crop_gen,
                                                   nullptr, 0, &libjpeg_image_mat);
    ASSERT_EQ(status, true);

    cv::Mat opencv_image_mat;
    std::string color_space("RGB");

    OpenCvPartialDecodeRandomCropImage(jpg.data(), jpg.size(), &opencv_random_crop_gen, color_space,
                                       opencv_image_mat);
    ImageUtil::ConvertColor("BGR", opencv_image_mat, color_space, opencv_image_mat);

    cv::Mat checkout = libjpeg_image_mat - opencv_image_mat;
    auto sum = cv::sum(cv::sum(checkout));
    ASSERT_EQ(sum[0], 0);
    // cv::imwrite("jpeg.ppm", libjpeg_image_mat);
    // cv::imwrite("opencv.ppm", opencv_image_mat);
  }
}

}  // namespace oneflow
