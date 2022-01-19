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
#ifndef _ONEFLOW_USER_KERNELS_ACTIVATION_KERNELS_H_
#define _ONEFLOW_USER_KERNELS_ACTIVATION_KERNELS_H_

#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

template<DeviceType device_type>
struct GenerateBaseGridImp {};

template<>
struct GenerateBaseGridImp<DeviceType::kCPU> {
  template<typename data_type>
  static void Linspace(std::vector<data_type>& grid, int64_t num_steps, bool align_corners) {
    if (num_steps <= 1) {
      for (auto& it : grid) { it = static_cast<data_type>(0.0); }
      return;
    }

    if (align_corners) {
      for (int i = 0; i < num_steps; i++) {
        grid[i] = static_cast<data_type>(-1.0 + 2.0 / (num_steps - 1) * i);
      }
    } else {
      for (int i = 0; i < num_steps; i++) {
        grid[i] = static_cast<data_type>((-1.0 + 2.0 / (num_steps - 1) * i) * (num_steps - 1)
                                         / num_steps);
      }
    }
  }

  template<typename data_type>
  static void Generate2D(user_op::KernelComputeContext*, data_type* grid_ptr, int64_t H, int64_t W,
                         bool align_corners) {
    std::vector<data_type> w_step(W);
    std::vector<data_type> h_step(H);
    Linspace(w_step, W, align_corners);
    Linspace(h_step, H, align_corners);

    for (int h = 0; h < H; h++) {
      data_type* row_ptr = grid_ptr + h * W * 3;
      for (int w = 0; w < W; w++) {
        data_type* pixel_ptr = row_ptr + w * 3;
        pixel_ptr[0] = w_step[w];
        pixel_ptr[1] = h_step[h];
        pixel_ptr[2] = static_cast<data_type>(1.0);
      }
    }
  }

  template<typename data_type>
  static void Generate3D(user_op::KernelComputeContext*, data_type* grid_ptr, int64_t D, int64_t H,
                         int64_t W, bool align_corners) {
    std::vector<data_type> w_step(W);
    std::vector<data_type> h_step(H);
    std::vector<data_type> d_step(D);
    Linspace(w_step, W, align_corners);
    Linspace(h_step, H, align_corners);
    Linspace(d_step, D, align_corners);

    for (int d = 0; d < D; d++) {
      data_type* image_ptr = grid_ptr + d * H * W * 4;
      for (int h = 0; h < H; h++) {
        data_type* row_ptr = image_ptr + h * W * 4;
        for (int w = 0; w < W; w++) {
          data_type* pixel_ptr = row_ptr + w * 4;
          pixel_ptr[0] = w_step[w];
          pixel_ptr[1] = h_step[h];
          pixel_ptr[2] = d_step[d];
          pixel_ptr[3] = static_cast<data_type>(1.0);
        }
      }
    }
  }
};

template<>
struct GenerateBaseGridImp<DeviceType::kCUDA> {
  static void Generate2D(user_op::KernelComputeContext* ctx, float* grid_ptr, int64_t H, int64_t W,
                         bool align_corners);
  static void Generate2D(user_op::KernelComputeContext* ctx, double* grid_ptr, int64_t H, int64_t W,
                         bool align_corners);

  static void Generate3D(user_op::KernelComputeContext* ctx, float* grid_ptr, int64_t D, int64_t H,
                         int64_t W, bool align_corners);
  static void Generate3D(user_op::KernelComputeContext* ctx, double* grid_ptr, int64_t D, int64_t H,
                         int64_t W, bool align_corners);
};

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ACTIVATION_KERNELS_H_
