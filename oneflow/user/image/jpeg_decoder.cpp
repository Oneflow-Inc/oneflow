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
#include <iostream>

#include "oneflow/user/image/jpeg_decoder.h"

namespace oneflow {

JpegDecoder::JpegDecoder() : cinfo_(), jerr_(), tmp_buf_() {}

JpegDecoder::~JpegDecoder() { jpeg_destroy_decompress(&cinfo_); }

JpegReturnType JpegDecoder::PartialDecode(const unsigned char* data, size_t length,
                                          RandomCropGenerator* random_crop_gen,
                                          unsigned char* workspace, size_t workspace_size,
                                          cv::Mat& out_mat) {
  cinfo_.err = jpeg_std_error(&jerr_);
  jpeg_create_decompress(&cinfo_);
  if (cinfo_.err->msg_code != 0) { return JpegReturnType::kError; }

  jpeg_mem_src(&cinfo_, data, length);
  if (cinfo_.err->msg_code != 0) { return JpegReturnType::kError; }

  int rc = jpeg_read_header(&cinfo_, TRUE);
  if (rc != 1) { return JpegReturnType::kError; }

  jpeg_start_decompress(&cinfo_);
  int width = cinfo_.output_width;
  int height = cinfo_.output_height;
  int pixel_size = cinfo_.output_components;

  unsigned char* crop_buf = nullptr;
  if (width * height * pixel_size > workspace_size) {
    tmp_buf_.resize(width * height * pixel_size);
    crop_buf = tmp_buf_.data();
  } else {
    crop_buf = workspace;
  }

  unsigned int u_crop_x = 0, u_crop_y = 0, u_crop_w = 0, u_crop_h = 0;
  if (random_crop_gen) {
    CropWindow crop;
    random_crop_gen->GenerateCropWindow({height, width}, &crop);
    u_crop_y = crop.anchor.At(0);
    u_crop_x = crop.anchor.At(1);
    u_crop_h = crop.shape.At(0);
    u_crop_w = crop.shape.At(1);
  } else {
    u_crop_y = 0;
    u_crop_x = 0;
    u_crop_h = height;
    u_crop_w = width;
  }

  unsigned int tmp_w = u_crop_w;
  jpeg_crop_scanline(&cinfo_, &u_crop_x, &tmp_w);
  int row_stride = tmp_w * pixel_size;
  if (jpeg_skip_scanlines(&cinfo_, u_crop_y) != u_crop_y) { return JpegReturnType::kError; }

  while (cinfo_.output_scanline < u_crop_y + u_crop_h) {
    unsigned char* buffer_array[1];
    buffer_array[0] = crop_buf + (cinfo_.output_scanline - u_crop_y) * row_stride;
    jpeg_read_scanlines(&cinfo_, buffer_array, 1);
  }

  jpeg_skip_scanlines(&cinfo_, cinfo_.output_height - u_crop_y - u_crop_h);
  jpeg_finish_decompress(&cinfo_);

  cv::Mat image(u_crop_h, tmp_w, CV_8UC3, crop_buf, cv::Mat::AUTO_STEP);
  cv::Rect roi;

  if (u_crop_w != tmp_w) {
    roi.x = tmp_w - u_crop_w;
    roi.y = 0;
    roi.width = u_crop_w;
    roi.height = u_crop_h;
    image(roi).copyTo(out_mat);
  } else {
    image.copyTo(out_mat);
  }

  return JpegReturnType::kOk;
}

}  // namespace oneflow
