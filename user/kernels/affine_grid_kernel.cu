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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "affine_grid_kernel.h"

namespace oneflow {

namespace {

template<typename data_type, bool align_corners>
OF_DEVICE_FUNC data_type LinspaceGPU(int32_t index, int32_t num_steps) {
  if (num_steps <= 1) { return static_cast<data_type>(0.0); }

  if (align_corners) {
    return static_cast<data_type>(-1.0 + 2.0 / (num_steps - 1) * index);
  } else {
    return static_cast<data_type>((-1.0 + 2.0 / (num_steps - 1) * index) * (num_steps - 1)
                                  / num_steps);
  }
}

template<typename data_type, bool align_corners>
__global__ void Generate2DBaseGridGPUKernel(const int32_t nthreads, data_type* grid_ptr, int32_t H,
                                            int32_t W) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int32_t h = index / W;
    const int32_t w = index % W;
    const int32_t pixel_length = 3;
    data_type* row_ptr = grid_ptr + h * W * pixel_length;
    data_type* pixel_ptr = row_ptr + w * pixel_length;
    data_type h_value = LinspaceGPU<data_type, align_corners>(h, H);
    data_type w_value = LinspaceGPU<data_type, align_corners>(w, W);

    pixel_ptr[0] = w_value;
    pixel_ptr[1] = h_value;
    pixel_ptr[2] = static_cast<data_type>(1.0);
  }
}

template<typename data_type, bool align_corners>
__global__ void Generate3DBaseGridGPUKernel(const int32_t nthreads, data_type* grid_ptr, int32_t D,
                                            int32_t H, int32_t W) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int32_t d = index / H;
    const int32_t h = index % H;
    const int32_t pixel_length = 4;
    data_type* image_ptr = grid_ptr + d * H * W * pixel_length;
    data_type* row_ptr = image_ptr + h * W * pixel_length;
    data_type d_value = LinspaceGPU<data_type, align_corners>(d, D);
    data_type h_value = LinspaceGPU<data_type, align_corners>(h, H);

    for (int32_t w = 0; w < W; ++w) {
      data_type* pixel_ptr = row_ptr + w * pixel_length;
      data_type w_value = LinspaceGPU<data_type, align_corners>(w, W);
      pixel_ptr[0] = w_value;
      pixel_ptr[1] = h_value;
      pixel_ptr[2] = d_value;
      pixel_ptr[3] = static_cast<data_type>(1.0);
    }
  }
}

}  // namespace

void GenerateBaseGridImp<DeviceType::kCUDA>::Generate2D(user_op::KernelComputeContext* ctx,
                                                        float* grid_ptr, int64_t H, int64_t W,
                                                        bool align_corners) {
  int count = H * W;
  if (align_corners) {
    RUN_CUDA_KERNEL((Generate2DBaseGridGPUKernel<float, true>), ctx->stream(), count, count,
                    grid_ptr, H, W);
  } else {
    RUN_CUDA_KERNEL((Generate2DBaseGridGPUKernel<float, false>), ctx->stream(), count, count,
                    grid_ptr, H, W);
  }
}
void GenerateBaseGridImp<DeviceType::kCUDA>::Generate2D(user_op::KernelComputeContext* ctx,
                                                        double* grid_ptr, int64_t H, int64_t W,
                                                        bool align_corners) {
  int count = H * W;
  if (align_corners) {
    RUN_CUDA_KERNEL((Generate2DBaseGridGPUKernel<double, true>), ctx->stream(), count, count,
                    grid_ptr, H, W);
  } else {
    RUN_CUDA_KERNEL((Generate2DBaseGridGPUKernel<double, false>), ctx->stream(), count, count,
                    grid_ptr, H, W);
  }
}

void GenerateBaseGridImp<DeviceType::kCUDA>::Generate3D(user_op::KernelComputeContext* ctx,
                                                        float* grid_ptr, int64_t D, int64_t H,
                                                        int64_t W, bool align_corners) {
  int count = D * H;
  if (align_corners) {
    RUN_CUDA_KERNEL((Generate3DBaseGridGPUKernel<float, true>), ctx->stream(), count, count,
                    grid_ptr, D, H, W);
  } else {
    RUN_CUDA_KERNEL((Generate3DBaseGridGPUKernel<float, false>), ctx->stream(), count, count,
                    grid_ptr, D, H, W);
  }
}

void GenerateBaseGridImp<DeviceType::kCUDA>::Generate3D(user_op::KernelComputeContext* ctx,
                                                        double* grid_ptr, int64_t D, int64_t H,
                                                        int64_t W, bool align_corners) {
  int count = D * H;
  if (align_corners) {
    RUN_CUDA_KERNEL((Generate3DBaseGridGPUKernel<double, true>), ctx->stream(), count, count,
                    grid_ptr, D, H, W);
  } else {
    RUN_CUDA_KERNEL((Generate3DBaseGridGPUKernel<double, false>), ctx->stream(), count, count,
                    grid_ptr, D, H, W);
  }
}

}  // namespace oneflow
