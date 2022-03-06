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
#include <cstddef>
#include <iostream>

#include "oneflow/user/image/jpeg_decoder.h"

namespace oneflow {

JpegDecoder::JpegDecoder() : compress_info_(), jpeg_err_(), decode_output_buf_() {
  compress_info_.err = jpeg_std_error(&jpeg_err_);
  jpeg_create_decompress(&compress_info_);
}

JpegDecoder::~JpegDecoder() { jpeg_destroy_decompress(&compress_info_); }

JpegReturnType JpegDecoder::PartialDecodeRandomCropImage(const unsigned char* data, size_t length,
                                                         RandomCropGenerator* random_crop_gen,
                                                         unsigned char* workspace,
                                                         size_t workspace_size, cv::Mat* out_mat) {
  if (compress_info_.err->msg_code != 0) { return JpegReturnType::kError; }

  jpeg_mem_src(&compress_info_, data, length);
  if (compress_info_.err->msg_code != 0) { return JpegReturnType::kError; }

  int rc = jpeg_read_header(&compress_info_, TRUE);
  if (rc != JPEG_HEADER_OK) { return JpegReturnType::kError; }

  jpeg_start_decompress(&compress_info_);
  int width = compress_info_.output_width;
  int height = compress_info_.output_height;
  int pixel_size = compress_info_.output_components;

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

  unsigned char* decode_output_buf = nullptr;
  size_t image_space_size = width * pixel_size;
  if (image_space_size > workspace_size) {
    decode_output_buf_.resize(image_space_size);
    decode_output_buf = decode_output_buf_.data();
  } else {
    decode_output_buf = workspace;
  }
  out_mat->create(u_crop_h, u_crop_w, CV_8UC3);

  unsigned int tmp_w = u_crop_w;
  jpeg_crop_scanline(&compress_info_, &u_crop_x, &tmp_w);
  if (jpeg_skip_scanlines(&compress_info_, u_crop_y) != u_crop_y) { return JpegReturnType::kError; }
  int row_offset = (tmp_w - u_crop_w) * pixel_size;
  int out_row_stride = u_crop_w * pixel_size;

  while (compress_info_.output_scanline < u_crop_y + u_crop_h) {
    unsigned char* buffer_array[1];
    buffer_array[0] = decode_output_buf;
    jpeg_read_scanlines(&compress_info_, buffer_array, 1);
    memcpy(out_mat->data + (compress_info_.output_scanline - u_crop_y) * out_row_stride,
           decode_output_buf + row_offset, out_row_stride);

    printf("line = %d \n", compress_info_.output_scanline);
  }

  jpeg_skip_scanlines(&compress_info_, height - u_crop_y - u_crop_h);
  jpeg_finish_decompress(&compress_info_);

  return JpegReturnType::kOk;
}

}  // namespace oneflow
