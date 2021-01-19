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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDTANH_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDTANH_KERNEL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
struct HardtanhFunctor {
  OF_DEVICE_FUNC explicit HardtanhFunctor(T min_val, T max_val)
      : min_val(min_val), max_val(max_val) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x <= min_val) {
      return min_val;
    } else if (x >= max_val) {
      return max_val;
    } else {
      return x;
    }
  }
  const T min_val;
  const T max_val;
};

template<typename T>
struct HardtanhGradFunctor {
  OF_DEVICE_FUNC explicit HardtanhGradFunctor(T min_val, T max_val)
      : min_val(min_val), max_val(max_val) {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const {
    return (y != min_val && y != max_val) ? dy : static_cast<T>(0);
  }
  const T min_val;
  const T max_val;
};

namespace {

template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseHardtanhFunctor final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, T min_val, T max_val, T* out,
                  const T* in);
};

template<DeviceType device_type, template<typename> class Opt, typename T>
struct ElemwiseHardtanhGradFunctor final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, T min_val, T max_val, T* dx, const T* y,
                  const T* dy);
};

}  // namespace

template<DeviceType device_type, template<typename> class Opt, typename T>
class ElemwiseHardtanhKernel final : public user_op::OpKernel {
 public:
  ElemwiseHardtanhKernel() = default;
  ~ElemwiseHardtanhKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T min_val = static_cast<T>(ctx->Attr<double>("min_val"));
    const T max_val = static_cast<T>(ctx->Attr<double>("max_val"));
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();
    ElemwiseHardtanhFunctor<device_type, Opt, T>()(ctx->device_ctx(), elem_cnt, min_val, max_val,
                                                   out_ptr, in_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, template<typename> class Opt, typename T>
class ElemwiseHardtanhGradKernel final : public user_op::OpKernel {
 public:
  ElemwiseHardtanhGradKernel() = default;
  ~ElemwiseHardtanhGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* y_ptr = y_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    const T min_val = static_cast<T>(ctx->Attr<double>("min_val"));
    const T max_val = static_cast<T>(ctx->Attr<double>("max_val"));
    const int64_t elem_cnt = y_tensor->shape().elem_cnt();
    ElemwiseHardtanhGradFunctor<device_type, Opt, T>()(ctx->device_ctx(), elem_cnt, min_val,
                                                       max_val, dx_ptr, y_ptr, dy_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_HARDTANH_KERNELS(device, dtype)                                                \
  REGISTER_USER_KERNEL("hardtanh")                                                              \
      .SetCreateFn<ElemwiseHardtanhKernel<device, HardtanhFunctor, dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });                                                                                       \
  REGISTER_USER_KERNEL("hardtanh_grad")                                                         \
      .SetCreateFn<ElemwiseHardtanhGradKernel<device, HardtanhGradFunctor, dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });
}  // namespace oneflow

#endif