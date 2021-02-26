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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
#include "oneflow/core/framework/framework.h"

#include "oneflow/core/common/data_type.h"

namespace oneflow {
template<DeviceType device_type, typename FunctorT, typename OutputT, typename InputA>
struct UnaryElemwiseXpuFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, OutputT* out, const InputA* input_a,
                  FunctorT functor);
};

template<typename FunctorT, typename OutputT, typename InputA>
struct UnaryElemwiseXpuFunctor<DeviceType::kCPU, FunctorT, OutputT, InputA> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, OutputT* out, const InputA* input_a,
                  FunctorT functor) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = functor(input_a[i]); }
  }
};

template<DeviceType device_type, typename FunctorT, typename OutputT, typename InputA,
         typename InputB>
struct BinaryElemwiseXpuFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, OutputT* out, const InputA* input_a,
                  const InputB* input_b, FunctorT functor);
};

template<typename FunctorT, typename OutputT, typename InputA, typename InputB>
struct BinaryElemwiseXpuFunctor<DeviceType::kCPU, FunctorT, OutputT, InputA, InputB> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, OutputT* out, const InputA* input_a,
                  const InputB* input_b, FunctorT functor) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = functor(input_a[i], input_b[i]); }
  }
};

template<DeviceType device_type, typename FunctorT, typename OutputT, typename InputA>
class UnaryElemwiseXpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryElemwiseXpuKernel);
  UnaryElemwiseXpuKernel() = default;
  ~UnaryElemwiseXpuKernel() = default;

  UnaryElemwiseXpuKernel(
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn,
      const std::string& output_name, const std::string& input_a_name)
      : FunctorCreateFn(FunctorCreateFn), output_name(output_name), input_a_name(input_a_name) {}

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_a_tensor = ctx->Tensor4ArgNameAndIndex(input_a_name, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);

    const ShapeView input_a_shape = input_a_tensor->shape();
    const ShapeView out_shape = out_tensor->shape();
    CHECK_EQ(input_a_shape, out_shape);

    const InputA* input_a_ptr = input_a_tensor->dptr<InputA>();
    OutputT* out_ptr = out_tensor->mut_dptr<OutputT>();
    const int64_t elem_cnt = input_a_shape.elem_cnt();

    UnaryElemwiseXpuFunctor<device_type, FunctorT, OutputT, InputA>()(
        ctx->device_ctx(), elem_cnt, out_ptr, input_a_ptr, FunctorCreateFn(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name;
  std::string input_a_name;
};

template<DeviceType device_type, typename FunctorT, typename OutputT, typename InputA,
         typename InputB>
class BinaryElemwiseXpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryElemwiseXpuKernel);
  BinaryElemwiseXpuKernel() = default;
  ~BinaryElemwiseXpuKernel() = default;

  BinaryElemwiseXpuKernel(
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn,
      const std::string& output_name, const std::string& input_a_name,
      const std::string& input_b_name)
      : FunctorCreateFn(FunctorCreateFn),
        output_name(output_name),
        input_a_name(input_a_name),
        input_b_name(input_b_name) {}

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_a_tensor = ctx->Tensor4ArgNameAndIndex(input_a_name, 0);
    const user_op::Tensor* input_b_tensor = ctx->Tensor4ArgNameAndIndex(input_b_name, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);

    const ShapeView input_a_shape = input_a_tensor->shape();
    const ShapeView input_b_shape = input_b_tensor->shape();
    const ShapeView out_shape = out_tensor->shape();
    CHECK_EQ(input_a_shape, out_shape);
    CHECK_EQ(input_b_shape, out_shape);

    const InputA* input_a_ptr = input_a_tensor->dptr<InputA>();
    const InputB* input_b_ptr = input_b_tensor->dptr<InputB>();
    OutputT* out_ptr = out_tensor->mut_dptr<OutputT>();
    const int64_t elem_cnt = input_a_shape.elem_cnt();

    BinaryElemwiseXpuFunctor<device_type, FunctorT, OutputT, InputA, InputB>()(
        ctx->device_ctx(), elem_cnt, out_ptr, input_a_ptr, input_b_ptr, FunctorCreateFn(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name;
  std::string input_a_name;
  std::string input_b_name;
};

#define REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITH_ATTR(                                             \
    device, kernel_name, functor, out_dtype, input_a_dtype, out_name, input_a_name, attr...)       \
  REGISTER_USER_KERNEL(kernel_name)                                                                \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                         \
        return new UnaryElemwiseXpuKernel<device, functor<out_dtype>, out_dtype, input_a_dtype>(   \
            [](user_op::KernelComputeContext* ctx) { return functor<out_dtype>(attr); }, out_name, \
            input_a_name);                                                                         \
      })                                                                                           \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceTag() == device)                                                      \
          & (user_op::HobDataType(input_a_name, 0) == GetDataType<out_dtype>::value));

#define REGISTER_BINARY_ELEMWISE_USER_KERNEL_WITH_ATTR(device, kernel_name, functor, out_dtype,    \
                                                       input_a_dtype, input_b_dtype, out_name,     \
                                                       input_a_name, input_b_name, attr...)        \
  REGISTER_USER_KERNEL(kernel_name)                                                                \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                         \
        return new BinaryElemwiseXpuKernel<device, functor<out_dtype>, out_dtype, input_a_dtype,   \
                                           input_b_dtype>(                                         \
            [](user_op::KernelComputeContext* ctx) { return functor<out_dtype>(attr); }, out_name, \
            input_a_name, input_b_name);                                                           \
      })                                                                                           \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceTag() == device)                                                      \
          & (user_op::HobDataType(input_a_name, 0) == GetDataType<out_dtype>::value));

#define REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITHOUT_ATTR(device, kernel_name, functor, out_dtype, \
                                                         input_a_dtype, out_name, input_a_name)   \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITH_ATTR(device, kernel_name, functor, out_dtype,          \
                                                input_a_dtype, out_name, input_a_name, )

#define REGISTER_BINARY_ELEMWISE_USER_KERNEL_WITHOUT_ATTR(device, kernel_name, functor, out_dtype, \
                                                          input_a_dtype, input_b_dtype, out_name,  \
                                                          input_a_name, input_b_name)              \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL_WITH_ATTR(device, kernel_name, functor, out_dtype,          \
                                                 input_a_dtype, input_b_dtype, out_name,           \
                                                 input_a_name, input_b_name, )

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
