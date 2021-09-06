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
#include <type_traits>
#include "oneflow/core/common/cplusplus_14.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/user/kernels/view_copy_kernel.h"

namespace oneflow {

namespace {

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

template<typename T, size_t ndim>
__global__ void copy_view(int64_t count, StrideParam<ndim> in_stride, StrideParam<ndim> out_stride,
                          const T* in_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, out_offset, count) {
    int64_t in_offset = out_stride.compute_offset(out_offset, in_stride);

    out_dptr[out_offset] = in_dptr[in_offset];
  }
}

template<typename T, size_t ndim>
void copy_view_wrapper(const DeviceCtx* ctx, int64_t count, const std::vector<int64_t>& in_stride,
                       const StrideVector& out_stride, const char* in_dptr, char* out_dptr) {
  StrideParam<ndim> param_in_stride(in_stride.data()), param_out_stride(out_stride.data());

  auto out_dptr_typed = reinterpret_cast<T*>(out_dptr);
  auto in_dptr_typed = reinterpret_cast<const T*>(in_dptr);

  copy_view<T, ndim>
      <<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          count, param_in_stride, param_out_stride, in_dptr_typed, out_dptr_typed);
}

template<typename T>
using copy_view_type = typename std::remove_reference<decltype(copy_view_wrapper<T, 1>)>::type*;

template<typename T, size_t... n>
struct copy_view_fn_map_type : std::unordered_map<size_t, copy_view_type<T>> {
  using base_type = std::unordered_map<size_t, copy_view_type<T>>;

  copy_view_fn_map_type() : base_type{{n + 1, copy_view_wrapper<T, n + 1>}...} {}

  copy_view_type<T> call(size_t ndim) {
    auto iter = base_type::find(ndim);

    if (iter != base_type::end()) {
      return iter->second;
    } else {
      UNIMPLEMENTED();

      return nullptr;
    }
  }
};

template<typename T, size_t... I>
copy_view_fn_map_type<T, I...> create_copy_view_fn_map(std::index_sequence<I...>) {
  return {};
}

template<typename T>
using copy_view_fn_map_t =
    decltype(create_copy_view_fn_map<T>(std::make_index_sequence<SHAPE_MAX_AXIS_SIZE>{}));
}  // namespace

template<typename T>
struct ViewCopyUtil<DeviceType::kGPU, T> : ViewCopyUtilBase {
  using ViewCopyUtilBase::ViewCopyUtilBase;

  static constexpr size_t dsize = sizeof(T);

  static copy_view_fn_map_t<T> copy_view_fn_map;

  void operator()() {
    if (contiguous_dim == -1) {
      OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, contiguous_block_size * dsize,
                                    cudaMemcpyDeviceToDevice, ctx->cuda_stream()));
    } else {
      const int64_t count = init_out_stride();

      const int ndim = in_shape.NumAxes();

      copy_view_fn_map.call(ndim)(ctx, count, in_stride, out_stride, in_dptr, out_dptr);
    }
  }
};

template<typename T>
copy_view_fn_map_t<T> ViewCopyUtil<DeviceType::kGPU, T>::copy_view_fn_map;

#define INSTANTIATE_VIEW_COPY_UTILS_FOR_GPU(T) template struct ViewCopyUtil<DeviceType::kGPU, T>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_VIEW_COPY_UTILS_FOR_GPU,
                     VIEW_COPY_TYPES VIEW_COPY_GPU_SPECIAL_TYPE)

}  // namespace oneflow
