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

#include <cstdint>
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

constexpr size_t NUM_DIM = 20;



template<typename T>
struct AsStridedFunctor final {
  void operator()(ep::Stream* stream, const T* input, T* output, const int64_t* out_shape,
                  const int32_t* stride, const int32_t out_ndim, const int32_t storage_offset,
                  const int32_t in_elem_cnt, const int32_t out_elem_cnt) {
    NdIndexOffsetHelper<int64_t, NUM_DIM> destIndexOffsetHelper(out_shape, out_ndim);
    FOR_RANGE(int64_t, i, 0, out_elem_cnt) {
      int64_t out_index[NUM_DIM];
      destIndexOffsetHelper.OffsetToNdIndex(i, out_index, out_ndim);
      int32_t in_offset = storage_offset;
      FOR_RANGE(int64_t, j, 0, out_ndim) { in_offset += out_index[j] * stride[j]; }
      output[i] = input[in_offset];
    }
  }
};

template<typename T>
struct AsStridedGradFunctor final {
  void operator()(ep::Stream* stream, const T* dy, T* dx, const int64_t* dy_shape,
                  const int32_t* stride, const int32_t dy_ndim, const int32_t storage_offset,
                  const int32_t dx_elem_cnt, const int32_t dy_elem_cnt) {
    NdIndexOffsetHelper<int64_t, NUM_DIM> destIndexOffsetHelper(dy_shape, dy_ndim);
    FOR_RANGE(int64_t, i, 0, dy_elem_cnt) {
      int64_t dy_index[NUM_DIM];
      destIndexOffsetHelper.OffsetToNdIndex(i, dy_index, dy_ndim);
      int32_t in_offset = storage_offset;
      FOR_RANGE(int64_t, j, 0, dy_ndim) { in_offset += dy_index[j] * stride[j]; }
      dx[in_offset] += dy[i];
    }
  }
};

}  // namespace

template<typename T>
class CpuAsStridedKernel final : public user_op::OpKernel {
 public:
  CpuAsStridedKernel() = default;
  ~CpuAsStridedKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto size = ctx->Attr<std::vector<int32_t>>("size");
    const auto stride = ctx->Attr<std::vector<int32_t>>("stride");
    const int32_t storage_offset = ctx->Attr<int32_t>("storage_offset");

    size_t out_ndim = output->shape().NumAxes();
    const int64_t* out_shape = output->shape().ptr();
    const size_t in_elem_cnt = input->shape().Count(0);
    const size_t out_elem_cnt = output->shape().Count(0);

    AsStridedFunctor<T>()(ctx->stream(), input->dptr<T>(), output->mut_dptr<T>(), out_shape,
                          stride.data(), out_ndim, storage_offset, in_elem_cnt, out_elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class CpuAsStridedGradKernel final : public user_op::OpKernel {
 public:
  CpuAsStridedGradKernel() = default;
  ~CpuAsStridedGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto size = ctx->Attr<std::vector<int32_t>>("size");
    const auto stride = ctx->Attr<std::vector<int32_t>>("stride");
    const int32_t storage_offset = ctx->Attr<int32_t>("storage_offset");

    size_t dy_ndim = dy->shape().NumAxes();
    const int64_t* dy_shape = dy->shape().ptr();
    const size_t dx_elem_cnt = dx->shape().Count(0);
    const size_t dy_elem_cnt = dy->shape().Count(0);

    Memset<DeviceType::kCPU>(ctx->stream(), dx->mut_dptr(), 0, dx->shape().Count(0) * sizeof(T));

    AsStridedGradFunctor<T>()(ctx->stream(), dy->dptr<T>(), dx->mut_dptr<T>(), dy_shape,
                              stride.data(), dy_ndim, storage_offset, dx_elem_cnt, dy_elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPUASSTRIDED_KERNEL(in_type)                                                 \
  REGISTER_USER_KERNEL("as_strided")                                                          \
      .SetCreateFn<CpuAsStridedKernel<in_type>>()                                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                         \
                       && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value)); \
  REGISTER_USER_KERNEL("as_strided_grad")                                                     \
      .SetCreateFn<CpuAsStridedGradKernel<in_type>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                         \
                       && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));

REGISTER_CPUASSTRIDED_KERNEL(float);
REGISTER_CPUASSTRIDED_KERNEL(double);
REGISTER_CPUASSTRIDED_KERNEL(bool);
REGISTER_CPUASSTRIDED_KERNEL(int8_t);
REGISTER_CPUASSTRIDED_KERNEL(int32_t);
REGISTER_CPUASSTRIDED_KERNEL(int64_t);

#undef REGISTER_CPUASSTRIDED_KERNEL

}  // namespace oneflow