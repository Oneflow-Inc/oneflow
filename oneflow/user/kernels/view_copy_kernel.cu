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
#include "oneflow/core/common/cplusplus_14.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/user/kernels/view_copy_kernel.h"

namespace oneflow {

namespace {

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

template<size_t N>
using uint_t = typename uint_type<N>::type;

template<size_t n>
struct StrideParam {
  int64_t stride[SHAPE_MAX_AXIS_SIZE];

  // NOLINTNEXTLINE
  StrideParam(const int64_t* input) {
    for (int i = 0; i < n; ++i) { stride[i] = input[i]; }
  }

  __device__ int64_t compute_offset(int64_t offset, const StrideParam& other) const {
    int64_t v = 0;

#pragma unroll
    for (int i = 0; i < n; ++i) {
      int64_t idx = offset / stride[i];
      v += idx * other.stride[i];
      offset -= idx * stride[i];
    }

    return v;
  }
};

template<size_t dsize, size_t ndim>
__global__ void copy_view(int64_t count, StrideParam<ndim> in_stride, StrideParam<ndim> out_stride,
                          const uint_t<dsize>* in_dptr, uint_t<dsize>* out_dptr) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, out_offset, count) {
    int64_t in_offset = out_stride.compute_offset(out_offset, in_stride);

    out_dptr[out_offset] = in_dptr[in_offset];
  }
}

template<size_t dsize, size_t ndim>
void copy_view_wrapper(const DeviceCtx* ctx, int64_t count, const std::vector<int64_t>& in_stride,
                       const StrideVector& out_stride, const char* in_dptr, char* out_dptr) {
  StrideParam<ndim> param_in_stride(in_stride.data()), param_out_stride(out_stride.data());

  auto out_dptr_typed = reinterpret_cast<uint_t<dsize>*>(out_dptr);
  auto in_dptr_typed = reinterpret_cast<const uint_t<dsize>*>(in_dptr);

  copy_view<dsize, ndim>
      <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          count, param_in_stride, param_out_stride, in_dptr_typed, out_dptr_typed);
}

using copy_view_type = decltype(copy_view_wrapper<1, 1>)*;

template<size_t... n>
struct copy_view_fn_map_type : std::unordered_map<std::pair<size_t, size_t>, copy_view_type> {
  using base_type = std::unordered_map<std::pair<size_t, size_t>, copy_view_type>;

  copy_view_fn_map_type()
      : base_type{{{1 << (n % 5), 1 + n / 5}, copy_view_wrapper<1 << (n % 5), 1 + n / 5>}...} {}

  copy_view_type call(size_t dsize, size_t ndim) {
    auto iter = find(std::make_pair(dsize, ndim));

    if (iter != end()) {
      return iter->second;
    } else {
      UNIMPLEMENTED();

      return nullptr;
    }
  }
};

template<size_t... I>
copy_view_fn_map_type<I...> create_copy_view_fn_map(std::index_sequence<I...>) {
  return {};
}

auto copy_view_fn_map = create_copy_view_fn_map(std::make_index_sequence<100>{});

}  // namespace

template<>
void ViewCopyUtil<kGPU>::operator()() {
  if (contiguous_dim == -1) {
    OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, contiguous_block_size * dsize,
                                  cudaMemcpyDeviceToDevice, ctx->cuda_stream()));
  } else {
    const int64_t count = init_out_stride();

    const int ndim = in_shape.NumAxes();

    copy_view_fn_map.call(dsize, ndim)(ctx, count, in_stride, out_stride, in_dptr, out_dptr);
  }
}

}  // namespace oneflow
