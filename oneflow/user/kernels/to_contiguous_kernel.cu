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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<size_t ndims>
struct StrideParam {
  int64_t stride[SHAPE_MAX_AXIS_SIZE];

  // NOLINTNEXTLINE
  StrideParam(const int64_t* stride_vec) {
    for (int i = 0; i < ndims; ++i) { stride[i] = stride_vec[i]; }
  }

  __device__ int64_t compute_index(int64_t output_idx, const StrideParam& input_params) const {
    int64_t input_idx = 0;

    #pragma unroll
    for (int i = 0; i < ndims; ++i) {
      int64_t idx = output_idx / stride[i];
      output_idx -= idx * stride[i];
      input_idx += idx * input_params.stride[i];
    }

    return input_idx;
  }
};

template<typename T, size_t ndim>
__global__ void to_contiguous(int64_t count, StrideParam<ndim> in_stride,
                              StrideParam<ndim> out_stride, const T* in_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, out_idx, count) {
    int64_t in_idx = out_stride.compute_index(out_idx, in_stride);

    out_dptr[out_idx] = in_dptr[in_idx];
  }
}

template<typename T, size_t ndim>
void to_contiguous_wrapper(ep::Stream* stream, int64_t count, const std::vector<int64_t>& in_stride,
                           const StrideVector& out_stride, const char* in_dptr, char* out_dptr) {
  StrideParam<ndim> param_in_stride(in_stride.data()), param_out_stride(out_stride.data());

  auto out_dptr_typed = reinterpret_cast<T*>(out_dptr);
  auto in_dptr_typed = reinterpret_cast<const T*>(in_dptr);

  to_contiguous<T, ndim><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, in_dptr_typed, out_dptr_typed);
}
template<typename T>
using to_contiguous_type =
    typename std::remove_reference<decltype(to_contiguous_wrapper<T, 1>)>::type*;
template<typename T, size_t... n>
struct to_contiguous_fn_map_type : std::unordered_map<size_t, to_contiguous_type<T>> {
  using base_type = std::unordered_map<size_t, to_contiguous_type<T>>;
  to_contiguous_fn_map_type() : base_type{{n + 1, to_contiguous_wrapper<T, n + 1>}...} {}
  to_contiguous_type<T> call(size_t ndim) {
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
to_contiguous_fn_map_type<T, I...> create_to_contiguous_fn_map(std::index_sequence<I...>) {
  return {};
}
template<typename T>
using to_contiguous_fn_map_t =
    decltype(create_to_contiguous_fn_map<T>(std::make_index_sequence<SHAPE_MAX_AXIS_SIZE>{}));
}  // namespace
template<typename T>
struct ToContiguousUtil<DeviceType::kCUDA, T> : ToContiguousUtilBase {
  using ToContiguousUtilBase::ToContiguousUtilBase;
  static constexpr size_t dsize = sizeof(T);
  static to_contiguous_fn_map_t<T> to_contiguous_fn_map;
  void operator()() {
    if (contiguous_dim == -1) {
      OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, block_size * dsize,
                                    cudaMemcpyDeviceToDevice,
                                    stream->As<ep::CudaStream>()->cuda_stream()));
    } else {
      const int ndim = in_shape.NumAxes();
      to_contiguous_fn_map.call(ndim)(stream, element_count, in_stride, out_stride, in_dptr, out_dptr);
    }
  }
};
template<typename T>
to_contiguous_fn_map_t<T> ToContiguousUtil<DeviceType::kCUDA, T>::to_contiguous_fn_map;
#define INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA(T) \
  template struct ToContiguousUtil<DeviceType::kCUDA, T>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA,
                     TO_CONTIGUOUS_TYPES TO_CONTIGUOUS_CUDA_SPECIAL_TYPE)
}  // namespace oneflow
