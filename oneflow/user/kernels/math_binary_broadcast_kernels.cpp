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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/ops/math_binary_broadcast_seq.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
namespace oneflow {

template<typename Context, ep::primitive::BinaryOp binary_op>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewBroadcastElementwiseBinaryPrimitive(
    Context* ctx) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* z = ctx->TensorDesc4ArgNameAndIndex("z", 0);
  size_t num_axes = z->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), binary_op, x->data_type(), z->data_type(), num_axes);
}

template<ep::primitive::BinaryOp binary_op>
class MathBinaryBroadcastEpKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  MathBinaryBroadcastEpKernel() = default;
  ~MathBinaryBroadcastEpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);

    auto primitive =
        NewBroadcastElementwiseBinaryPrimitive<user_op::KernelComputeContext, binary_op>(ctx);
    CHECK(primitive.get() != nullptr) << "Exceeds maximum supported dimensions";

    const int64_t x_elem_cnt = x->shape().elem_cnt();
    const int64_t y_elem_cnt = y->shape().elem_cnt();
    size_t num_src0_dims = x->shape().NumAxes();
    size_t num_src1_dims = y->shape().NumAxes();

    int64_t zero_dim = 1;
    int64_t* src0_dims = const_cast<int64_t*>(x->shape().ptr());
    int64_t* src1_dims = const_cast<int64_t*>(y->shape().ptr());

    if (x_elem_cnt != 0 && y_elem_cnt != 0) {
      if (num_src0_dims == 0) {
        num_src0_dims = 1;
        src0_dims = &zero_dim;
      }
      if (num_src1_dims == 0) {
        num_src1_dims = 1;
        src1_dims = &zero_dim;
      }

      primitive->Launch(ctx->stream(), num_src0_dims, src0_dims, x->dptr(), num_src1_dims,
                        src1_dims, y->dptr(), z->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<ep::primitive::BinaryOp binary_op>
auto MathBinaryBroadcastPrimitiveExists() {
  return hob::make_custom("MathBinaryBroadcastPrimitiveExists", [](const user_op::KernelRegContext&
                                                                       ctx) {
    return NewBroadcastElementwiseBinaryPrimitive<const user_op::KernelRegContext, binary_op>(&ctx).
    operator bool();
  });
}

#define REGISTER_BINARY_BROADCAST_EP_KERNEL(math_type_pair, binary_op) \
  REGISTER_USER_KERNEL(math_type_pair)                                 \
      .SetCreateFn<MathBinaryBroadcastEpKernel<binary_op>>()           \
      .SetIsMatchedHob(MathBinaryBroadcastPrimitiveExists<binary_op>() == true);

REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_add", ep::primitive::BinaryOp::kAdd)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_sub", ep::primitive::BinaryOp::kSub)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_mul", ep::primitive::BinaryOp::kMul)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_div", ep::primitive::BinaryOp::kDiv)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_minimum", ep::primitive::BinaryOp::kMin)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_maximum", ep::primitive::BinaryOp::kMax)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_pow", ep::primitive::BinaryOp::kPow)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_equal", ep::primitive::BinaryOp::kEqual)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_not_equal", ep::primitive::BinaryOp::kNotEqual)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_greater", ep::primitive::BinaryOp::kGreaterThan)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_greater_equal",
                                    ep::primitive::BinaryOp::kGreaterEqual)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_less", ep::primitive::BinaryOp::kLessThan)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_less_equal", ep::primitive::BinaryOp::kLessEqual)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_logical_and", ep::primitive::BinaryOp::kLogicalAnd)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_logical_or", ep::primitive::BinaryOp::kLogicalOr)
REGISTER_BINARY_BROADCAST_EP_KERNEL("broadcast_logical_xor", ep::primitive::BinaryOp::kLogicalXor)

template<DeviceType device_type, typename T, typename K,
         void (*binary_func)(ep::Stream* stream, const XpuVarNdarray<K>& z,
                             const XpuVarNdarray<const T>& x, const XpuVarNdarray<const T>& y)>
class MathBinaryBroadcastKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MathBinaryBroadcastKernel() = default;
  ~MathBinaryBroadcastKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
    const T* dptr_x = tensor_x->dptr<T>();
    const T* dptr_y = tensor_y->dptr<T>();
    K* dptr_z = tensor_z->mut_dptr<K>();
    size_t num_axes = tensor_z->shape().NumAxes();
    binary_func(ctx->stream(), XpuVarNdarray<K>(tensor_z->shape(), dptr_z, num_axes),
                XpuVarNdarray<const T>(tensor_x->shape(), dptr_x, num_axes),
                XpuVarNdarray<const T>(tensor_y->shape(), dptr_y, num_axes));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define MATH_BINARY_BROADCAST_DEFAULT_FUNC_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_floor_mod", FloorMod) \
  OF_PP_MAKE_TUPLE_SEQ("broadcast_fmod", FMod)

#define REGISTER_MATH_BINARY_BROADCAST_KERNEL(math_type_pair, device, data_type_pair) \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                              \
      .SetCreateFn<MathBinaryBroadcastKernel<                                         \
          device, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(data_type_pair), \
          &NdarrayUtil<device, OF_PP_PAIR_FIRST(data_type_pair)>::OF_PP_CAT(          \
              Broadcast, OF_PP_PAIR_SECOND(math_type_pair))>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                           \
                       && (user_op::HobDataType("z", 0) == OF_PP_PAIR_SECOND(data_type_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_MATH_BINARY_BROADCAST_KERNEL, MATH_BINARY_BROADCAST_DEFAULT_FUNC_SEQ, DEVICE_TYPE_SEQ,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
// gpu half
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
                                 MATH_BINARY_BROADCAST_DEFAULT_FUNC_SEQ, (DeviceType::kCUDA),
                                 FLOAT16_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
