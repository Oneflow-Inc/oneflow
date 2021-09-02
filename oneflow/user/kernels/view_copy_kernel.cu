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

  // NOLINTNEXTLINE
  StrideParam(const int64_t* input, size_t n) {
    for (int i = 0; i < n; ++i) { stride[i] = input[i]; }
  }

  __device__ int64_t compute_offset(int64_t offset, int ndim, const StrideParam& other) const {
    int64_t v = 0;

#pragma unroll
    for (int i = 0; i < ndim; ++i) {
      int64_t idx = offset / stride[i];
      v += idx * other.stride[i];
      offset -= idx * stride[i];
    }

    return v;
  }
};

template<size_t N>
struct uint_type;

template<>
struct uint_type<1> {
  using type = uint8_t;
};

template<>
struct uint_type<2> {
  using type = uint16_t;
};

template<>
struct uint_type<4> {
  using type = uint32_t;
};

template<>
struct uint_type<8> {
  using type = uint64_t;
};

template<>
struct uint_type<16> {
  using type = ulonglong2;
};

template<size_t dsize>
__global__ void copy_view(int64_t count, StrideParam in_stride, StrideParam out_stride,
                          const uint_type<dsize>* in_dptr, uint_type<dsize>* out_dptr,
                          int ndim) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, out_offset, count) {
    out_dptr[out_offset] = in_dptr[out_stride.compute_offset(out_offset, ndim, in_stride)];
  }
}

template<size_t dsize>
void copy_view_wrapper(const DeviceCtx* ctx, int64_t count, const std::vector<int64_t>& in_stride,
                       const StrideVector& out_stride, const char* in_dptr, char* out_dptr,
                       int ndim) {
  StrideParam param_in_stride(in_stride.data(), ndim), param_out_stride(out_stride.data(), ndim);

  auto out_dptr_typed = reinterpret_cast<uint_type<dsize>*>(out_dptr);
  auto in_dptr_typed = reinterpret_cast<const uint_type<dsize>*>(in_dptr);

  copy_view<dsize><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, in_dptr_typed, out_dptr_typed, ndim);
}

using copy_view_type = decltype(copy_view_wrapper<1>)*;

template<size_t... dsizes>
struct copy_view_fn_map_type : std::unordered_map<size_t, copy_view_type> {
  using base_type = std::unordered_map<size_t, copy_view_type>;

  copy_view_fn_map_type() : base_type{{1 << dsizes, copy_view_wrapper<1 << dsizes>}...} {}

  copy_view_type call(size_t n) {
    auto iter = find(n);

    if (iter != end()) {
      return iter->second;
    } else {
      UNIMPLEMENTED();
    }
  }
};

copy_view_fn_map_type<0, 1, 2, 3, 4> copy_view_fn_map;

}  // namespace

template<>
void ViewCopyUtil<kGPU>::operator()() {
  if (contiguous_dim == -1) {
    OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, contiguous_block_size * dsize,
                                  cudaMemcpyDeviceToDevice, ctx->cuda_stream()));
  } else {
    const int64_t count = init_out_stride();

    const int ndim = in_shape.NumAxes();

    copy_view_fn_map.call(dsize)(ctx, count, in_stride, out_stride, in_dptr, out_dptr, ndim);
  }
}

}  // namespace oneflow
