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
#include "oneflow/user/image/image_util.h"

namespace oneflow {

class LibjpegCtx {
 public:
  explicit LibjpegCtx(struct jpeg_decompress_struct* compress_info)
      : compress_info_(compress_info) {}
  ~LibjpegCtx() { jpeg_destroy_decompress(compress_info_); }
  OF_DISALLOW_COPY_AND_MOVE(LibjpegCtx);
  struct jpeg_decompress_struct* compress_info() {
    return compress_info_;
  }

 private:
  struct jpeg_decompress_struct* compress_info_;
};

bool JpegPartialDecodeRandomCropImage(const unsigned char* data, size_t length,
                                      RandomCropGenerator* random_crop_gen,
                                      unsigned char* workspace, size_t workspace_size,
                                      cv::Mat* out_mat) {
  struct jpeg_decompress_struct compress_info {};
  struct jpeg_error_mgr jpeg_err {};
  compress_info.err = jpeg_std_error(&jpeg_err);
  jpeg_create_decompress(&compress_info);
  if (compress_info.err->msg_code != 0) { return false; }

  LibjpegCtx ctx_guard(&compress_info);

  jpeg_mem_src(ctx_guard.compress_info(), data, length);
  if (ctx_guard.compress_info()->err->msg_code != 0) { return false; }

  int rc = jpeg_read_header(ctx_guard.compress_info(), TRUE);
  if (rc != JPEG_HEADER_OK) { return false; }

  jpeg_start_decompress(ctx_guard.compress_info());
  int width = ctx_guard.compress_info()->output_width;
  int height = ctx_guard.compress_info()->output_height;
  int pixel_size = ctx_guard.compress_info()->output_components;

  unsigned int u_crop_x = 0, u_crop_y = 0, u_crop_w = width, u_crop_h = height;
  if (random_crop_gen) {
    CropWindow crop;
    random_crop_gen->GenerateCropWindow({height, width}, &crop);
    u_crop_y = crop.anchor.At(0);
    u_crop_x = crop.anchor.At(1);
    u_crop_h = crop.shape.At(0);
    u_crop_w = crop.shape.At(1);
  }

  unsigned int tmp_w = u_crop_w;
  jpeg_crop_scanline(ctx_guard.compress_info(), &u_crop_x, &tmp_w);
  if (jpeg_skip_scanlines(ctx_guard.compress_info(), u_crop_y) != u_crop_y) { return false; }

  int row_offset = (tmp_w - u_crop_w) * pixel_size;
  int out_row_stride = u_crop_w * pixel_size;
  std::vector<unsigned char> decode_output_buf;
  unsigned char* decode_output_pointer = nullptr;
  size_t image_space_size = width * pixel_size;

  if (image_space_size > workspace_size) {
    decode_output_buf.resize(image_space_size);
    decode_output_pointer = decode_output_buf.data();
  } else {
    decode_output_pointer = workspace;
  }
  out_mat->create(u_crop_h, u_crop_w, CV_8UC3);

  while (ctx_guard.compress_info()->output_scanline < u_crop_y + u_crop_h) {
    unsigned char* buffer_array[1];
    buffer_array[0] = decode_output_pointer;
    unsigned int read_line_index = ctx_guard.compress_info()->output_scanline;
    jpeg_read_scanlines(ctx_guard.compress_info(), buffer_array, 1);
    memcpy(out_mat->data + (read_line_index - u_crop_y) * out_row_stride,
           decode_output_pointer + row_offset, out_row_stride);
  }

  jpeg_skip_scanlines(ctx_guard.compress_info(), height - u_crop_y - u_crop_h);
  jpeg_finish_decompress(ctx_guard.compress_info());

  return true;
}

void OpenCvPartialDecodeRandomCropImage(const unsigned char* data, size_t length,
                                        RandomCropGenerator* random_crop_gen,
                                        const std::string& color_space, cv::Mat& out_mat) {
  cv::Mat image =
      cv::imdecode(cv::Mat(1, length, CV_8UC1, const_cast<unsigned char*>(data)),
                   ImageUtil::IsColor(color_space) ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
  int W = image.cols;
  int H = image.rows;

  // random crop
  if (random_crop_gen != nullptr) {
    CHECK(image.data != nullptr);
    cv::Mat image_roi;
    CropWindow crop;
    random_crop_gen->GenerateCropWindow({H, W}, &crop);
    const int y = crop.anchor.At(0);
    const int x = crop.anchor.At(1);
    const int newH = crop.shape.At(0);
    const int newW = crop.shape.At(1);
    CHECK(newW > 0 && newW <= W);
    CHECK(newH > 0 && newH <= H);
    cv::Rect roi(x, y, newW, newH);
    image(roi).copyTo(out_mat);
    W = out_mat.cols;
    H = out_mat.rows;
    CHECK(W == newW);
    CHECK(H == newH);
  } else {
    image.copyTo(out_mat);
  }
}

}  // namespace oneflow
