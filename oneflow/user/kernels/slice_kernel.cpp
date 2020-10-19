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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/slice_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

SliceParams ConstructSliceParams(user_op::KernelComputeContext* ctx, const user_op::Tensor* entire,
                                 const user_op::Tensor* sliced) {
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  const int64_t ndim = entire->shape().NumAxes();
  CHECK_LE(ndim, kSliceMaxDims);
  CHECK_EQ(sliced->shape().NumAxes(), ndim);
  CHECK_EQ(start_vec.size(), ndim);
  CHECK_EQ(stop_vec.size(), ndim);
  CHECK_EQ(step_vec.size(), ndim);

  SliceParams params;
  std::memset(&params, 0, sizeof(SliceParams));
  params.ndim = ndim;
  FOR_RANGE(int, i, 0, params.ndim) {
    const int64_t dim_size = entire->shape().At(i);
    const int64_t slice_size = sliced->shape().At(i);
    const int64_t step = step_vec.at(i);
    CHECK_NE(step, 0);
    const int64_t start = RegulateSliceStart(start_vec.at(i), dim_size);
    const int64_t stop = RegulateSliceStop(stop_vec.at(i), dim_size);
    if (step > 0) {
      CHECK_LT(start + step * (slice_size - 1), stop);
    } else {
      CHECK_GT(start + step * (slice_size - 1), stop);
    }
    params.dims[i] = dim_size;
    params.start[i] = start;
    params.step[i] = step;
    params.size[i] = slice_size;
  }
  return params;
}

}  // namespace

template<DeviceType device_type, typename T>
class SliceKernel final : public user_op::OpKernel {
 public:
  SliceKernel() = default;
  ~SliceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    SliceParams params = ConstructSliceParams(ctx, x_tensor, y_tensor);
    SliceKernelUtil<device_type, T>::Forward(ctx->device_ctx(), params, x_tensor->dptr<T>(),
                                             y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class SliceGradKernel final : public user_op::OpKernel {
 public:
  SliceGradKernel() = default;
  ~SliceGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t dx_byte_size = dx_tensor->shape().elem_cnt() * sizeof(T);
    Memset<device_type>(ctx->device_ctx(), dx_tensor->mut_dptr<T>(), 0, dx_byte_size);
    SliceParams params = ConstructSliceParams(ctx, dx_tensor, dy_tensor);
    SliceKernelUtil<device_type, T>::Backward(ctx->device_ctx(), params, dy_tensor->dptr<T>(),
                                              dx_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class SliceUpdateKernel final : public user_op::OpKernel {
 public:
  SliceUpdateKernel() = default;
  ~SliceUpdateKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* update_tensor = ctx->Tensor4ArgNameAndIndex("update", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    Memcpy<device_type>(ctx->device_ctx(), y_tensor->mut_dptr<T>(), x_tensor->dptr<T>(),
                        y_tensor->shape().elem_cnt() * sizeof(T));
    SliceParams params = ConstructSliceParams(ctx, y_tensor, update_tensor);
    SliceKernelUtil<device_type, T>::Backward(ctx->device_ctx(), params, update_tensor->dptr<T>(),
                                              y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SLICE_KERNELS(device, dtype)                                                   \
  REGISTER_USER_KERNEL("slice").SetCreateFn<SliceKernel<device, dtype>>().SetIsMatchedHob(      \
      (user_op::HobDeviceTag() == device)                                                       \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));                           \
  REGISTER_USER_KERNEL("slice_grad")                                                            \
      .SetCreateFn<SliceGradKernel<device, dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));         \
  REGISTER_USER_KERNEL("slice_update")                                                          \
      .SetCreateFn<SliceUpdateKernel<device, dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobDataType("update", 0) == GetDataType<dtype>::value))      \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_SLICE_KERNELS_WITH_DEVICE(device) \
  REGISTER_SLICE_KERNELS(device, float)            \
  REGISTER_SLICE_KERNELS(device, double)           \
  REGISTER_SLICE_KERNELS(device, int32_t)          \
  REGISTER_SLICE_KERNELS(device, int64_t)          \
  REGISTER_SLICE_KERNELS(device, int8_t)           \
  REGISTER_SLICE_KERNELS(device, uint8_t)

REGISTER_SLICE_KERNELS_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_SLICE_KERNELS_WITH_DEVICE(DeviceType::kGPU)
REGISTER_SLICE_KERNELS(DeviceType::kGPU, float16)
#endif

}  // namespace oneflow
