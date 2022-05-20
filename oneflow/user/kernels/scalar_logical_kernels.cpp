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
#include "oneflow/user/kernels/scalar_logical_kernels.h"

namespace oneflow {

template<template<typename T> class BIN_OP, typename T>
struct ScalarLogicalFunctor<DeviceType::kCPU, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in,
                  bool* out) {
    DoScalarLogical<BIN_OP, T>(elem_cnt, scalar, in, out);
  }
};

template<template<typename> class UnaryFunctor, typename T>
void DoScalarLogicalWithStride(const int64_t elem_cnt, const StrideParam& in_stride,
                               const StrideParam& out_stride, const T scalar, const T* in,
                               bool* out) {
  for (int64_t i = 0; i < elem_cnt; ++i) {
    const int64_t in_idx = compute_index(i, in_stride, out_stride);
    out[i] = UnaryFunctor<T>::Invoke(in[in_idx], scalar);
  }
}

template<template<typename T> class BIN_OP, typename T>
struct ScalarLogicalWithStrideFunctor<DeviceType::kCPU, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const StrideParam& in_stride,
                  const StrideParam& out_stride, const T scalar, const T* in, bool* out) {
    DoScalarLogicalWithStride<BIN_OP, T>(elem_cnt, in_stride, out_stride, scalar, in, out);
  }
};

template<DeviceType device_type, template<typename> class BIN_OP, typename T>
class ScalarLogicalKernel final : public user_op::OpKernel {
 public:
  ScalarLogicalKernel() = default;
  ~ScalarLogicalKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const T* in_ptr = in->dptr<T>();
    bool* out_ptr = out->mut_dptr<bool>();

    // compute is_contiguous and construct input/output stride params
    const size_t ndim = in->stride().NumAxes();
    const DimVector& in_stride_vec = in->stride().StrideVec();
    const DimVector& out_stride_vec = out->stride().StrideVec();
    DimVector in_shape_vec;
    in->shape().ToDimVector(&in_shape_vec);
    bool is_contiguous = oneflow::one::IsContiguous(in_shape_vec, in_stride_vec);
    StrideParam in_stride(in_stride_vec.data(), ndim), out_stride(out_stride_vec.data(), ndim);

    int64_t elem_cnt = out->shape().elem_cnt();
    if (elem_cnt != 0) {
      if (is_contiguous) {
        ScalarLogicalFunctor<device_type, BIN_OP, T>()(ctx->stream(), elem_cnt, scalar_operand,
                                                       in_ptr, out_ptr);
      } else {
        ScalarLogicalWithStrideFunctor<device_type, BIN_OP, T>()(
            ctx->stream(), elem_cnt, in_stride, out_stride, scalar_operand, in_ptr, out_ptr);
      }
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op,       \
                                                           input_dtype_pair)                     \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn<ScalarLogicalKernel<device, binary_op, OF_PP_PAIR_FIRST(input_dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(input_dtype_pair)));

#define REGISTER_SCALAR_LOGICAL_KERNEL(device, dtype_pair)                                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_equal", BinaryFuncEQ, \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_not_equal",           \
                                                     BinaryFuncNE, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_greater",             \
                                                     BinaryFuncGT, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_greater_equal",       \
                                                     BinaryFuncGE, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_less", BinaryFuncLT,  \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_less_equal",          \
                                                     BinaryFuncLE, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_or", BinaryFuncOR,    \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_xor", BinaryFuncXOR,  \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_and", BinaryFuncAND,  \
                                                     dtype_pair);

// we register bool, uint8_t, int8_t, int32_t, int64_t, float, double.
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_LOGICAL_KERNEL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_LOGICAL_KERNEL, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

}  // namespace oneflow
