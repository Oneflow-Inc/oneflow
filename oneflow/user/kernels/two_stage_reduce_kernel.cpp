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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/kernels/two_stage_reduce_kernel_util.h"

namespace oneflow {
namespace {

template<DeviceType device_type, typename T, typename U>
struct CopyTensor;

template<typename T, typename U>
struct CopyTensor<DeviceType::kCPU, T, U> {
  static void Call(DeviceCtx* ctx, const int64_t n, const T* src, U* dst) { CopyElem(src, dst, n); }
};

template<typename T, typename U>
struct CopyTensor<DeviceType::kGPU, T, U> {
  static void Call(DeviceCtx* ctx, const int64_t n, const T* src, U* dst) {
#ifdef WITH_CUDA
    CopyElemOnGpu(ctx, src, dst, n);
#else
    UNIMPLEMENTED();
#endif
  }
};

}  // namespace

namespace user_op {

template<template<typename> class BinaryFunc, DeviceType device_type, typename T>
class ReduceDeviceStageKernel final : public OpKernel {
 public:
  ReduceDeviceStageKernel() = default;
  ~ReduceDeviceStageKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* count = ctx->Tensor4ArgNameAndIndex("count", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* reduce_tmp_buf = tmp_buffer->mut_dptr<T>();
    int32_t* mask_tmp_buf = tmp_buffer->mut_dptr<int32_t>();
    const size_t tmp_bytes =
        GetCudaAlignedSize(in->shape().elem_cnt() * std::max(sizeof(T), sizeof(int32_t)));
    int32_t* reduce_sum_tmp_buf =
        reinterpret_cast<int32_t*>(tmp_buffer->mut_dptr<char>() + tmp_bytes);

    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(out->shape(), out->mut_dptr<T>()),
        XpuVarNdarray<const T>(in->shape(), in->dptr<T>()),
        XpuVarNdarray<T>(in->shape(), reduce_tmp_buf));
    NdarrayUtil<device_type, T>::BroadcastEQ(
        ctx->device_ctx(), XpuVarNdarray<int8_t>(mask->shape(), mask->mut_dptr<int8_t>()),
        XpuVarNdarray<const T>(in->shape(), in->dptr<T>()),
        XpuVarNdarray<const T>(out->shape(), out->dptr<T>()));

    CopyTensor<device_type, int8_t, int32_t>::Call(ctx->device_ctx(), mask->shape().elem_cnt(),
                                                   mask->dptr<int8_t>(), mask_tmp_buf);
    NdarrayUtil<device_type, int32_t>::ReduceSum(
        ctx->device_ctx(), XpuVarNdarray<int32_t>(count->shape(), count->mut_dptr<int32_t>()),
        XpuVarNdarray<const int32_t>(mask->shape(), mask_tmp_buf),
        XpuVarNdarray<int32_t>(mask->shape(), reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenDeviceStageInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
    const size_t tmp_bytes =
        GetCudaAlignedSize(in_shape->elem_cnt() * std::max(sizeof(T), sizeof(int32_t)));
    const size_t reduce_sum_tmp_bytes = GetCudaAlignedSize(in_shape->elem_cnt() * sizeof(int32_t));
    return tmp_bytes + reduce_sum_tmp_bytes;
  };
}

#define REGISTER_REDUCE_DEVICE_STAGE_KERNEL(op_name, binary_func, device, dtype_pair)            \
  REGISTER_USER_KERNEL(op_name)                                                                  \
      .SetCreateFn<ReduceDeviceStageKernel<binary_func, device, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInferTmpSizeFn(GenDeviceStageInferTmpSizeFn<OF_PP_PAIR_FIRST(dtype_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_DEVICE_STAGE_KERNEL, ("reduce_max_device_stage"),
                                 (BinaryFuncMax), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_DEVICE_STAGE_KERNEL, ("reduce_min_device_stage"),
                                 (BinaryFuncMin), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)

template<DeviceType device_type, typename T>
class ReduceDeviceStageGradKernel final : public OpKernel {
 public:
  ReduceDeviceStageGradKernel() = default;
  ~ReduceDeviceStageGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const user_op::Tensor* out_diff = ctx->Tensor4ArgNameAndIndex("out_diff", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* count = ctx->Tensor4ArgNameAndIndex("count", 0);
    user_op::Tensor* in_diff = ctx->Tensor4ArgNameAndIndex("in_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* tmp_buf_ptr = tmp_buffer->mut_dptr<T>();
    const size_t tmp_bytes = GetCudaAlignedSize(out_diff->shape().elem_cnt() * sizeof(T));
    T* broadcasted_tmp_buf_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + tmp_bytes);

    TwoStageReduceKernelUtil<device_type, T, int32_t>::Divide(
        ctx->device_ctx(), out_diff->shape().elem_cnt(), out_diff->dptr<T>(),
        count->dptr<int32_t>(), tmp_buf_ptr);

    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(in_diff->shape(), broadcasted_tmp_buf_ptr),
        XpuVarNdarray<const T>(out_diff->shape(), tmp_buf_ptr));

    TwoStageReduceKernelUtil<device_type, T, int8_t>::Mask(
        ctx->device_ctx(), in_diff->shape().elem_cnt(), broadcasted_tmp_buf_ptr,
        mask->dptr<int8_t>(), in_diff->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenDeviceStageGradInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape* out_diff_shape = ctx->Shape4ArgNameAndIndex("out_diff", 0);
    const Shape* in_diff_shape = ctx->Shape4ArgNameAndIndex("in_diff", 0);
    const size_t tmp_bytes = GetCudaAlignedSize(out_diff_shape->elem_cnt() * sizeof(T));
    const size_t broadcasted_tmp_bytes = GetCudaAlignedSize(in_diff_shape->elem_cnt() * sizeof(T));
    return tmp_bytes + broadcasted_tmp_bytes;
  };
}

#define REGISTER_REDUCE_DEVICE_STAGE_GRAD_KERNEL(op_name, device, dtype_pair)                   \
  REGISTER_USER_KERNEL(op_name)                                                                 \
      .SetCreateFn<ReduceDeviceStageGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("in_diff", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn(GenDeviceStageGradInferTmpSizeFn<OF_PP_PAIR_FIRST(dtype_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_DEVICE_STAGE_GRAD_KERNEL,
                                 ("reduce_max_device_stage_grad"), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_DEVICE_STAGE_GRAD_KERNEL,
                                 ("reduce_min_device_stage_grad"), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)

template<template<typename> class BinaryFunc, DeviceType device_type, typename T>
class ReduceGlobalStageKernel final : public OpKernel {
 public:
  ReduceGlobalStageKernel() = default;
  ~ReduceGlobalStageKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape = CreateReducedShape(in->shape(), {axis.begin(), axis.end()});
    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, out->mut_dptr<T>()),
        XpuVarNdarray<const T>(in->shape(), in->dptr<T>()),
        XpuVarNdarray<T>(in->shape(), tmp_buffer->mut_dptr<T>()));

    NdarrayUtil<device_type, T>::BroadcastEQ(
        ctx->device_ctx(), XpuVarNdarray<int8_t>(in->shape(), mask->mut_dptr<int8_t>()),
        XpuVarNdarray<const T>(in->shape(), in->dptr<T>()),
        XpuVarNdarray<const T>(reduced_shape, out->dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_GLOBAL_STAGE_KERNEL(op_name, binary_func, device, dtype_pair)            \
  REGISTER_USER_KERNEL(op_name)                                                                  \
      .SetCreateFn<ReduceGlobalStageKernel<binary_func, device, OF_PP_PAIR_FIRST(dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInferTmpSizeFn([](InferContext* ctx) {                                                 \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                             \
        return in_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair));                      \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_GLOBAL_STAGE_KERNEL, ("reduce_max_global_stage"),
                                 (BinaryFuncMax), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_GLOBAL_STAGE_KERNEL, ("reduce_min_global_stage"),
                                 (BinaryFuncMin), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)

template<DeviceType device_type, typename T>
class ReduceGlobalStageGradKernel final : public OpKernel {
 public:
  ReduceGlobalStageGradKernel() = default;
  ~ReduceGlobalStageGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const user_op::Tensor* out_diff = ctx->Tensor4ArgNameAndIndex("out_diff", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* device_count = ctx->Tensor4ArgNameAndIndex("device_count", 0);
    user_op::Tensor* in_diff = ctx->Tensor4ArgNameAndIndex("in_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int32_t* device_count_with_mask = tmp_buffer->mut_dptr<int32_t>();
    const size_t device_count_with_mask_bytes =
        GetCudaAlignedSize(device_count->shape().elem_cnt() * sizeof(int32_t));
    int32_t* global_count =
        reinterpret_cast<int32_t*>(tmp_buffer->mut_dptr<char>() + device_count_with_mask_bytes);
    const size_t global_count_bytes =
        GetCudaAlignedSize(out_diff->shape().elem_cnt() * sizeof(int32_t));
    int32_t* reduce_sum_tmp_buf = reinterpret_cast<int32_t*>(
        tmp_buffer->mut_dptr<char>() + device_count_with_mask_bytes + global_count_bytes);
    const size_t reduce_sum_tmp_bytes =
        GetCudaAlignedSize(device_count->shape().elem_cnt() * sizeof(int32_t));
    T* divided_buf_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + device_count_with_mask_bytes
                             + global_count_bytes + reduce_sum_tmp_bytes);
    const size_t divided_buf_bytes = GetCudaAlignedSize(out_diff->shape().elem_cnt() * sizeof(T));
    T* broadcasted_divided_buf_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + device_count_with_mask_bytes
                             + global_count_bytes + reduce_sum_tmp_bytes + divided_buf_bytes);

    TwoStageReduceKernelUtil<device_type, int32_t, int8_t>::Mask(
        ctx->device_ctx(), device_count->shape().elem_cnt(), device_count->dptr<int32_t>(),
        mask->dptr<int8_t>(), device_count_with_mask);

    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        CreateReducedShape(device_count->shape(), {axis.begin(), axis.end()});

    NdarrayUtil<device_type, int32_t>::ReduceSum(
        ctx->device_ctx(), XpuVarNdarray<int32_t>(reduced_shape, global_count),
        XpuVarNdarray<const int32_t>(device_count->shape(), device_count_with_mask),
        XpuVarNdarray<int32_t>(device_count->shape(), reduce_sum_tmp_buf));

    TwoStageReduceKernelUtil<device_type, T, int32_t>::Divide(
        ctx->device_ctx(), out_diff->shape().elem_cnt(), out_diff->dptr<T>(), global_count,
        divided_buf_ptr);

    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(in_diff->shape(), broadcasted_divided_buf_ptr),
        XpuVarNdarray<const T>(out_diff->shape(), divided_buf_ptr));

    TwoStageReduceKernelUtil<device_type, T, int32_t>::Scale(
        ctx->device_ctx(), in_diff->shape().elem_cnt(), broadcasted_divided_buf_ptr,
        device_count_with_mask, in_diff->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenGlobalStageGradInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const Shape* device_count_shape = ctx->Shape4ArgNameAndIndex("device_count", 0);
    const Shape* out_diff_shape = ctx->Shape4ArgNameAndIndex("out_diff", 0);
    const Shape* in_diff_shape = ctx->Shape4ArgNameAndIndex("in_diff", 0);
    const size_t device_count_with_mask_bytes =
        GetCudaAlignedSize(device_count_shape->elem_cnt() * sizeof(int32_t));
    const size_t global_count_bytes =
        GetCudaAlignedSize(out_diff_shape->elem_cnt() * sizeof(int32_t));
    const size_t reduce_sum_tmp_bytes =
        GetCudaAlignedSize(device_count_shape->elem_cnt() * sizeof(int32_t));
    const size_t divided_buf_bytes = GetCudaAlignedSize(out_diff_shape->elem_cnt() * sizeof(T));
    const size_t broadcasted_divided_buf_bytes =
        GetCudaAlignedSize(in_diff_shape->elem_cnt() * sizeof(T));
    const size_t total_bytes = device_count_with_mask_bytes + global_count_bytes
                               + reduce_sum_tmp_bytes + divided_buf_bytes
                               + broadcasted_divided_buf_bytes;
    return total_bytes;
  };
}

#define REGISTER_REDUCE_GLOBAL_STAGE_GRAD_KERNEL(op_name, device, dtype_pair)                   \
  REGISTER_USER_KERNEL(op_name)                                                                 \
      .SetCreateFn<ReduceGlobalStageGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("in_diff", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn(GenGlobalStageGradInferTmpSizeFn<OF_PP_PAIR_FIRST(dtype_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_GLOBAL_STAGE_GRAD_KERNEL,
                                 ("reduce_max_global_stage_grad"), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_GLOBAL_STAGE_GRAD_KERNEL,
                                 ("reduce_min_global_stage_grad"), DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
