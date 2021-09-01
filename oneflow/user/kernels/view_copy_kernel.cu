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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/user/kernels/view_copy_kernel.h"

namespace oneflow {

namespace {

struct StrideParam {
  int64_t stride[SHAPE_MAX_AXIS_SIZE];
  
  //NOLINTNEXTLINE
  StrideParam(const int64_t* input, size_t n) {
    for (int i = 0; i < n; ++i) {
      stride[i] = input[i];
    }
  }

  __device__ void compute_index(int64_t offset, int ndim, int64_t index[]) const {
    int64_t v = offset;
  
  #pragma unroll
    for (int i = 0; i < ndim; ++i) {
      int64_t idx = v / stride[i];
      index[i] = idx;
      v -= idx * stride[i];
    }
  }
  
  __device__ int64_t compute_offset(const int64_t index[], int ndim) const {
    int64_t v = 0;
  
  #pragma unroll
    for (int i = 0; i < ndim; ++i) { v += index[i] * stride[i]; }
  
    return v;
  }
};

__global__ void copy_view(int64_t count, size_t dsize,
                          StrideParam in_stride, StrideParam out_stride,
                          const char* in_dptr, char* out_dptr, int64_t ndim) {
  int64_t in_index[SHAPE_MAX_AXIS_SIZE];

  CUDA_1D_KERNEL_LOOP_T(int64_t, out_offset, count) {

    out_stride.compute_index(out_offset, ndim, in_index);
    const int64_t in_offset = in_stride.compute_offset(in_index, ndim);

    char *out_dptr_offset = out_dptr + out_offset * dsize;
    const char *in_dptr_offset = in_dptr + in_offset * dsize;

#pragma unroll
    for (int j = 0; j < dsize; ++j) { out_dptr_offset[j] = in_dptr_offset[j]; }
  }
}

}  // namespace

template<>
void ViewCopyUtil<kGPU>::operator()() {
  if (contiguous_dim == -1) {
    OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, contiguous_block_size * dsize,
                                  cudaMemcpyDeviceToDevice, ctx->cuda_stream()));
  } else {
    const int64_t count = init_out_stride();

    const int ndim = in_shape.NumAxes();

    StrideParam param_in_stride(in_stride.data(), ndim), param_out_stride(out_stride.data(), ndim);

    copy_view<<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        count, dsize, param_in_stride, param_out_stride, in_dptr, out_dptr, ndim);
  }
}

}  // namespace oneflow
