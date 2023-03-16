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

#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

enum class BinaryOpMLU {
  kAdd,
  kMul,
  kSub,
};

union Param {
  int int_value;
  float float_value;
};

struct TransformParams {
  Param alpha;  // scaling factor of tensor input
  Param beta;   // bias factor of tensor input
};

template<BinaryOpMLU, typename T>
struct Alpha;

template<typename T>
struct Alpha<BinaryOpMLU::kAdd, T> {
  T operator()(Scalar value) { return T(1); }
};

template<typename T>
struct Alpha<BinaryOpMLU::kMul, T> {
  T operator()(Scalar value) { return value.Value<T>(); }
};

template<typename T>
struct Alpha<BinaryOpMLU::kSub, T> {
  T operator()(Scalar value) { return T(1); }
};

template<BinaryOpMLU, typename T>
struct Beta;

template<typename T>
struct Beta<BinaryOpMLU::kAdd, T> {
  T operator()(Scalar value) { return value.Value<T>(); }
};

template<typename T>
struct Beta<BinaryOpMLU::kMul, T> {
  T operator()(Scalar value) { return T(0); }
};

template<typename T>
struct Beta<BinaryOpMLU::kSub, T> {
  T operator()(Scalar value) { return -value.Value<T>(); }
};

// If the data type of tensors is float or half, the data type of alpha and beta should be
// `float*`. If the data type of tensors is int32, the data type of alpha and beta should be
// `int*`.
template<BinaryOpMLU op, typename T, typename std::enable_if<IsFloating<T>::value>::type* = nullptr>
void SetTransformParams(Scalar src0, TransformParams& params) {
  params.alpha.float_value = Alpha<op, float>()(src0);
  params.beta.float_value = Beta<op, float>()(src0);
}

template<BinaryOpMLU op, typename T, typename std::enable_if<IsIntegral<T>::value>::type* = nullptr>
void SetTransformParams(Scalar src0, TransformParams& params) {
  params.alpha.int_value = Alpha<op, int32_t>()(src0);
  params.beta.int_value = Beta<op, int32_t>()(src0);
}

template<BinaryOpMLU op, typename T>
void LaunchMathKernel(user_op::KernelComputeContext* ctx, Scalar src0, const user_op::Tensor* in,
                      user_op::Tensor* out) {
  auto num_axes = in->shape_view().NumAxes();
  CHECK(num_axes <= CNNL_DIM_MAX) << "The number of dimensions is no more than CNNL_DIM_MAX ("
                                  << num_axes << " <= " << CNNL_DIM_MAX << ")";
  TransformParams params;
  SetTransformParams<op, T>(src0, params);
  CnnlTensorDescriptor input_desc;
  input_desc.set(in);
  auto handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
  OF_CNNL_CHECK(cnnlTransform(handle, &params.alpha, input_desc.desc(), in->dptr(), &params.beta,
                              out->mut_dptr()));
}

}  // namespace

template<BinaryOpMLU op, typename T>
class ScalarMathKernelMLU final : public user_op::OpKernel {
 public:
  ScalarMathKernelMLU() = default;
  ~ScalarMathKernelMLU() = default;

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
      const bool is_add_sub_0 =
          (op == BinaryOpMLU::kAdd || op == BinaryOpMLU::kSub) && value.Value<double>() == 0.0;
      // TODO(Jianhua Zheng): support kDiv
      const bool is_mul_div_1 = (op == BinaryOpMLU::kMul) && value.Value<double>() == 1.0;
      if ((is_add_sub_0 || is_mul_div_1) && in->dptr() == out->dptr()) { return; }
      LaunchMathKernel<op, T>(ctx, value, in, out);
    } else {
      // For 0-d Tensor
      return;
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_MATH_USER_KERNEL(op_name, binary_op, dtype)                         \
  REGISTER_USER_KERNEL(op_name)                                                             \
      .SetCreateFn<ScalarMathKernelMLU<binary_op, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                       \
                       && (user_op::HobDataType("in", 0) == user_op::HobDataType("out", 0)) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))     \
      .SetInplaceProposalFn(                                                                \
          [](const user_op::InferContext& ctx,                                              \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {        \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));               \
            return Maybe<void>::Ok();                                                       \
          });

REGISTER_SCALAR_MATH_USER_KERNEL("scalar_add", BinaryOpMLU::kAdd, float)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_add", BinaryOpMLU::kAdd, float16)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_add", BinaryOpMLU::kAdd, int32_t)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_mul", BinaryOpMLU::kMul, float)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_mul", BinaryOpMLU::kMul, float16)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_mul", BinaryOpMLU::kMul, int32_t)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_sub", BinaryOpMLU::kSub, float)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_sub", BinaryOpMLU::kSub, float16)
REGISTER_SCALAR_MATH_USER_KERNEL("scalar_sub", BinaryOpMLU::kSub, int32_t)

}  // namespace oneflow
