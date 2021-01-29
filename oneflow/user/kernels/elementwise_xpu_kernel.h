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
template<DeviceType device_type, typename FunctorT, typename T>
struct UnaryElemwiseXpuFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in, FunctorT functor);
};

template<typename FunctorT, typename T>
struct UnaryElemwiseXpuFunctor<DeviceType::kCPU, FunctorT, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in, FunctorT functor) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = functor(in[i]); }
  }
};

template<DeviceType device_type, typename FunctorT, typename T>
struct BinaryElemwiseXpuFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in_1, const T* in_2,
                  FunctorT functor);
};

template<typename FunctorT, typename T>
struct BinaryElemwiseXpuFunctor<DeviceType::kCPU, FunctorT, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in_1, const T* in_2,
                  FunctorT functor) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = functor(in_1[i], in_2[i]); }
  }
};

template<DeviceType device_type, typename FunctorT, typename T>
struct TernaryElemwiseXpuFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in_1, const T* in_2,
                  const T* in_3, FunctorT functor);
};

template<typename FunctorT, typename T>
struct TernaryElemwiseXpuFunctor<DeviceType::kCPU, FunctorT, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, T* out, const T* in_1, const T* in_2,
                  const T* in_3, FunctorT functor) {
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out[i] = functor(in_1[i], in_2[i], in_3[i]); }
  }
};

#define INSTANTIATE_UNARY_XPU_FUNCTOR(device, functor, T) \
  template struct UnaryElemwiseXpuFunctor<device, functor<T>, T>;

#define INSTANTIATE_BINARY_XPU_FUNCTOR(device, functor, T) \
  template struct BinaryElemwiseXpuFunctor<device, functor<T>, T>;

#define INSTANTIATE_TERNARY_XPU_FUNCTOR(device, functor, T) \
  template struct TernaryElemwiseXpuFunctor<device, functor<T>, T>;

template<DeviceType device_type, typename FunctorT, typename T>
class UnaryElemwiseXpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryElemwiseXpuKernel);
  UnaryElemwiseXpuKernel() = default;
  ~UnaryElemwiseXpuKernel() = default;

  UnaryElemwiseXpuKernel(
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn,
      const std::string& output_name, const std::string& input_name)
      : FunctorCreateFn(FunctorCreateFn), output_name(output_name), input_name(input_name) {}

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex(input_name, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();

    UnaryElemwiseXpuFunctor<device_type, FunctorT, T>()(ctx->device_ctx(), elem_cnt, out_ptr,
                                                        in_ptr, FunctorCreateFn(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name;
  std::string input_name;
};

template<DeviceType device_type, typename FunctorT, typename T>
class BinaryElemwiseXpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryElemwiseXpuKernel);
  BinaryElemwiseXpuKernel() = default;
  ~BinaryElemwiseXpuKernel() = default;

  BinaryElemwiseXpuKernel(
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn,
      const std::string& output_name, const std::string& input_name1,
      const std::string& input_name2)
      : FunctorCreateFn(FunctorCreateFn),
        output_name(output_name),
        input_name1(input_name1),
        input_name2(input_name2) {}

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor1 = ctx->Tensor4ArgNameAndIndex(input_name1, 0);
    const user_op::Tensor* in_tensor2 = ctx->Tensor4ArgNameAndIndex(input_name2, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);
    const T* in_ptr1 = in_tensor1->dptr<T>();
    const T* in_ptr2 = in_tensor2->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor1->shape().elem_cnt();

    BinaryElemwiseXpuFunctor<device_type, FunctorT, T>()(ctx->device_ctx(), elem_cnt, out_ptr,
                                                         in_ptr1, in_ptr2, FunctorCreateFn(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name;
  std::string input_name1;
  std::string input_name2;
};

template<DeviceType device_type, typename FunctorT, typename T>
class TernaryElemwiseXpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TernaryElemwiseXpuKernel);
  TernaryElemwiseXpuKernel() = default;
  ~TernaryElemwiseXpuKernel() = default;

  TernaryElemwiseXpuKernel(
      std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn,
      const std::string& output_name, const std::string& input_name1,
      const std::string& input_name2, const std::string& input_name3)
      : FunctorCreateFn(FunctorCreateFn),
        output_name(output_name),
        input_name1(input_name1),
        input_name2(input_name2),
        input_name3(input_name3) {}

  std::function<FunctorT(user_op::KernelComputeContext* ctx)> FunctorCreateFn;  // The functor

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor1 = ctx->Tensor4ArgNameAndIndex(input_name1, 0);
    const user_op::Tensor* in_tensor2 = ctx->Tensor4ArgNameAndIndex(input_name2, 0);
    const user_op::Tensor* in_tensor3 = ctx->Tensor4ArgNameAndIndex(input_name3, 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex(output_name, 0);
    const T* in_ptr1 = in_tensor1->dptr<T>();
    const T* in_ptr2 = in_tensor2->dptr<T>();
    const T* in_ptr3 = in_tensor3->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const int64_t elem_cnt = in_tensor1->shape().elem_cnt();

    TernaryElemwiseXpuFunctor<device_type, FunctorT, T>()(
        ctx->device_ctx(), elem_cnt, out_ptr, in_ptr1, in_ptr2, in_ptr3, FunctorCreateFn(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name;
  std::string input_name1;
  std::string input_name2;
  std::string input_name3;
};

}  // namespace oneflow
#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
