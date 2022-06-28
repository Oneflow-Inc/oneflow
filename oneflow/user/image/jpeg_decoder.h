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
#ifndef ONEFLOW_USER_IMAGE_JPEG_DECODER_H_
#define ONEFLOW_USER_IMAGE_JPEG_DECODER_H_
#include <jpeglib.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "oneflow/user/image/random_crop_generator.h"

namespace oneflow {

bool JpegPartialDecodeRandomCropImage(const unsigned char* data, size_t length,
                                      RandomCropGenerator* random_crop_gen,
                                      unsigned char* workspace, size_t workspace_size,
                                      cv::Mat* out_mat);

void OpenCvPartialDecodeRandomCropImage(const unsigned char* data, size_t length,
                                        RandomCropGenerator* random_crop_gen,
                                        const std::string& color_space, cv::Mat& out_mat);

}  // namespace oneflow
#endif  // ONEFLOW_USER_IMAGE_JPEG_DECODER_H_
