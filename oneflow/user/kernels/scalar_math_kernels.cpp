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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewBroadcastElementwiseBinaryPrimitive(
    Context* ctx, ep::primitive::BinaryOp op) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  const int64_t ndims = y->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), y->data_type(), ndims);
}

template<ep::primitive::BinaryOp op>
auto BroadcastElementwiseBinaryPrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseBinaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewBroadcastElementwiseBinaryPrimitive(&ctx, op).operator bool();
                          });
}

}  // namespace

template<ep::primitive::BinaryOp op>
class ScalarMathKernel final : public user_op::OpKernel {
 public:
  ScalarMathKernel() = default;
  ~ScalarMathKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = Scalar(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = Scalar(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), in->shape_view().NumAxes(), in->shape_view().ptr(),
                        in->dptr(), value, out->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<ep::primitive::BinaryOp op>
class ScalarReverseMathKernel final : public user_op::OpKernel {
 public:
  ScalarReverseMathKernel() = default;
  ~ScalarReverseMathKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = Scalar(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = Scalar(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), value, in->shape_view().NumAxes(), in->shape_view().ptr(),
                        in->dptr(), out->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define SCALAR_MATH_SEQ                                                       \
  OF_PP_MAKE_TUPLE_SEQ("scalar_add", ep::primitive::BinaryOp::kAdd)           \
  OF_PP_MAKE_TUPLE_SEQ("scalar_mul", ep::primitive::BinaryOp::kMul)           \
  OF_PP_MAKE_TUPLE_SEQ("scalar_div", ep::primitive::BinaryOp::kDiv)           \
  OF_PP_MAKE_TUPLE_SEQ("scalar_floordiv", ep::primitive::BinaryOp::kFloorDiv) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_fmod", ep::primitive::BinaryOp::kFmod)         \
  OF_PP_MAKE_TUPLE_SEQ("scalar_pow", ep::primitive::BinaryOp::kPow)

#define REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(op_name, binary_op)                 \
  REGISTER_USER_KERNEL(op_name).SetCreateFn<ScalarMathKernel<binary_op>>().SetIsMatchedHob( \
      (BroadcastElementwiseBinaryPrimitiveExists<binary_op>()));

OF_PP_FOR_EACH_TUPLE(REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL, SCALAR_MATH_SEQ)

#define REGISTER_UNARY_MATH_SCALAR_REVERSE_ELEMWISE_USER_KERNEL(op_name, binary_op)                \
  REGISTER_USER_KERNEL(op_name).SetCreateFn<ScalarReverseMathKernel<binary_op>>().SetIsMatchedHob( \
      (BroadcastElementwiseBinaryPrimitiveExists<binary_op>()));

REGISTER_UNARY_MATH_SCALAR_REVERSE_ELEMWISE_USER_KERNEL("scalar_reverse_pow",
                                                        ep::primitive::BinaryOp::kPow)

template<DeviceType device_type, typename T>
class CpuScalarPowGradKernel final : public user_op::OpKernel {
 public:
  CpuScalarPowGradKernel() = default;
  ~CpuScalarPowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }

    const int32_t elem_cnt = x_tensor->shape_view().elem_cnt();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] =
          scalar_operand * (std::pow(x_ptr[i], scalar_operand - static_cast<T>(1))) * dy_ptr[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(device, dtype)  \
  REGISTER_USER_KERNEL("scalar_pow_grad")                   \
      .SetCreateFn<CpuScalarPowGradKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(DeviceType::kCPU, double);

template<DeviceType device_type, typename T>
class CpuScalarReversePowGradKernel final : public user_op::OpKernel {
 public:
  CpuScalarReversePowGradKernel() = default;
  ~CpuScalarReversePowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }

    const int32_t elem_cnt = x_tensor->shape_view().elem_cnt();
    // NOTE: y = a^x    ==>>   dy/dx = a^x * lna
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = std::pow(scalar_operand, x_ptr[i]) * std::log(scalar_operand) * dy_ptr[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_REVERSE_POW_GRAD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("scalar_reverse_pow_grad")                  \
      .SetCreateFn<CpuScalarReversePowGradKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)        \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_REVERSE_POW_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_REVERSE_POW_GRAD_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
