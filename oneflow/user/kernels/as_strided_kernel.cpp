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

#include "oneflow/core/common/util.h"
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
  void operator()(ep::Stream* stream, const T* input_buf, T* output_buf, const int64_t* dest_dims,
                  const int64_t* stride, const int64_t dest_num_dims, const int64_t storage_offset,
                  const int64_t input_num, const int64_t output_num) {
    NdIndexOffsetHelper<int64_t, NUM_DIM> destIndexOffsetHelper(dest_dims, dest_num_dims);
    FOR_RANGE(int64_t, i, 0, output_num) {
      int64_t dst_index[NUM_DIM];
      destIndexOffsetHelper.OffsetToNdIndex(i, dst_index, dest_num_dims);
      int64_t index_in_input = storage_offset;
      FOR_RANGE(int64_t, j, 0, dest_num_dims) { index_in_input += dst_index[j] * stride[j]; }
      output_buf[i] = input_buf[index_in_input];
    }
  }
};

template<typename T>
struct AsStridedGradFunctor final {
  void operator()(ep::Stream* stream, const T* dy_buf, T* dx_buf, const int64_t* dy_dims,
                  const int64_t* stride, const int64_t dy_num_dims, const int64_t storage_offset,
                  const int64_t dx_num, const int64_t dy_num) {
    NdIndexOffsetHelper<int64_t, NUM_DIM> destIndexOffsetHelper(dy_dims, dy_num_dims);
    FOR_RANGE(int64_t, i, 0, dy_num) {
      int64_t dy_index[NUM_DIM];
      destIndexOffsetHelper.OffsetToNdIndex(i, dy_index, dy_num_dims);
      int64_t index_in_dx = storage_offset;
      FOR_RANGE(int64_t, j, 0, dy_num_dims) { index_in_dx += dy_index[j] * stride[j]; }
      dx_buf[index_in_dx] += dy_buf[i];
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
    const auto size = ctx->Attr<std::vector<int64_t>>("size");
    const auto stride = ctx->Attr<std::vector<int64_t>>("stride");
    const int64_t storage_offset = ctx->Attr<int64_t>("storage_offset");

    size_t dest_num_dims = output->shape_view().NumAxes();
    const int64_t* dest_dims = output->shape_view().ptr();
    const size_t input_num = input->shape_view().Count(0);
    const size_t output_num = output->shape_view().Count(0);

    AsStridedFunctor<T>()(ctx->stream(), input->dptr<T>(), output->mut_dptr<T>(), dest_dims,
                          stride.data(), dest_num_dims, storage_offset, input_num, output_num);
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
    const auto size = ctx->Attr<std::vector<int64_t>>("size");
    const auto stride = ctx->Attr<std::vector<int64_t>>("stride");
    const int64_t storage_offset = ctx->Attr<int64_t>("storage_offset");

    size_t dy_num_dims = dy->shape_view().NumAxes();
    const int64_t* dy_dims = dy->shape_view().ptr();
    const size_t dx_num = dx->shape_view().Count(0);
    const size_t dy_num = dy->shape_view().Count(0);

    Memset<DeviceType::kCPU>(ctx->stream(), dx->mut_dptr(), 0,
                             dx->shape_view().Count(0) * sizeof(T));

    AsStridedGradFunctor<T>()(ctx->stream(), dy->dptr<T>(), dx->mut_dptr<T>(), dy_dims,
                              stride.data(), dy_num_dims, storage_offset, dx_num, dy_num);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_ASSTRIDED_KERNEL(in_type)                        \
  REGISTER_USER_KERNEL("as_strided")                                  \
      .SetCreateFn<CpuAsStridedKernel<in_type>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));

REGISTER_CPU_ASSTRIDED_KERNEL(float);
REGISTER_CPU_ASSTRIDED_KERNEL(double);
REGISTER_CPU_ASSTRIDED_KERNEL(int8_t);
REGISTER_CPU_ASSTRIDED_KERNEL(uint8_t);
REGISTER_CPU_ASSTRIDED_KERNEL(int32_t);
REGISTER_CPU_ASSTRIDED_KERNEL(int64_t);

#undef REGISTER_CPU_ASSTRIDED_KERNEL

#define REGISTER_CPU_ASSTRIDED_GRAD_KERNEL(in_type)                   \
  REGISTER_USER_KERNEL("as_strided_grad")                             \
      .SetCreateFn<CpuAsStridedGradKernel<in_type>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));

REGISTER_CPU_ASSTRIDED_GRAD_KERNEL(float);
REGISTER_CPU_ASSTRIDED_GRAD_KERNEL(double);

#undef REGISTER_CPU_ASSTRIDED_GRAD_KERNEL

REGISTER_USER_KERNEL("as_strided")
    .SetCreateFn<CpuAsStridedKernel<bool>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     && (user_op::HobDataType("input", 0) == GetDataType<bool>::value));

}  // namespace oneflow
