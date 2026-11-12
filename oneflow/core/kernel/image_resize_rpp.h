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
#ifndef ONEFLOW_CORE_KERNEL_IMAGE_RESIZE_RPP_H_
#define ONEFLOW_CORE_KERNEL_IMAGE_RESIZE_RPP_H_

#if defined(WITH_CUDA) && defined(USE_HIP)

#include <hip/hip_runtime.h>
#include <memory>

namespace oneflow {

// Bilinear crop-and-resize of an interleaved 8-bit RGB image on a HIP stream, with RPP. RPP is
// used only from its own translation unit: rpp.h and miopen.h both declare an unscoped enumerator
// named BILINEAR, so rpp.h cannot be included next to the cuDNN headers.
class RppImageResizer final {
 public:
  // Returns nullptr if RPP cannot be set up on the stream.
  static std::unique_ptr<RppImageResizer> Create(hipStream_t stream);
  ~RppImageResizer();
  RppImageResizer(const RppImageResizer&) = delete;
  RppImageResizer& operator=(const RppImageResizer&) = delete;

  // Resizes the roi (x, y, w, h) of src to dst; returns false if RPP reports an error.
  bool ResizeRgb(const unsigned char* src, int src_width, int src_height, int roi_x, int roi_y,
                 int roi_w, int roi_h, unsigned char* dst, int dst_width, int dst_height);

 private:
  struct Impl;
  explicit RppImageResizer(Impl* impl);
  Impl* impl_;
};

}  // namespace oneflow

#endif  // defined(WITH_CUDA) && defined(USE_HIP)

#endif  // ONEFLOW_CORE_KERNEL_IMAGE_RESIZE_RPP_H_
