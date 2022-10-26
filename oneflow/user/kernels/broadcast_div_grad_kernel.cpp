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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {

namespace {

template<DeviceType device, typename T>
class BroadcastDivGradKernel final : public user_op::OpKernel {
 public:
  BroadcastDivGradKernel() = default;
  ~BroadcastDivGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* z_tensor = ctx->Tensor4ArgNameAndIndex("z", 0);
    const user_op::Tensor* dz_tensor = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const int64_t num_axes = dz_tensor->shape_view().NumAxes();
    XpuVarNdarray<const T> dz(dz_tensor->shape_view(), dz_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    auto bcast_div = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
        ctx->device_type(), ep::primitive::BinaryOp::kDiv, z_tensor->data_type(),
        z_tensor->data_type(), z_tensor->shape_view().NumAxes());
    CHECK(bcast_div);
    bcast_div->Launch(ctx->stream(), z_tensor->shape_view().NumAxes(), z_tensor->shape_view().ptr(),
                      z_tensor->dptr(), y_tensor->shape_view().NumAxes(),
                      y_tensor->shape_view().ptr(), y_tensor->dptr<T>(), tmp_buffer->mut_dptr<T>());

    auto bcast_mul = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
        ctx->device_type(), ep::primitive::BinaryOp::kMul, dz_tensor->data_type(),
        dz_tensor->data_type(), dz_tensor->shape_view().NumAxes());
    CHECK(bcast_mul);
    bcast_mul->Launch(ctx->stream(), dz_tensor->shape_view().NumAxes(),
                      dz_tensor->shape_view().ptr(), tmp_buffer->dptr(),
                      dz_tensor->shape_view().NumAxes(), dz_tensor->shape_view().ptr(),
                      dz_tensor->dptr<T>(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device, T>::ReduceSum(
        ctx->stream(),
        XpuVarNdarray<T>(dy_tensor->shape_view(), dy_tensor->mut_dptr<T>(), num_axes), const_tmp,
        tmp);

    auto negative = ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
        ctx->device_type(), ep::primitive::UnaryOp::kNegative, dy_tensor->data_type(),
        dy_tensor->data_type());
    CHECK(negative);
    negative->Launch(ctx->stream(), dy_tensor->dptr(), dy_tensor->mut_dptr(),
                     dy_tensor->shape_view().elem_cnt());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BROADCAST_DIV_GRAD_KERNEL(device, dtype_pair)                             \
  REGISTER_USER_KERNEL("broadcast_div_grad")                                               \
      .SetCreateFn<BroadcastDivGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                         \
        const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);                       \
        DataType data_type = z.data_type();                                                \
        const int64_t elem_cnt = z.shape().elem_cnt();                                     \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));                \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_DIV_GRAD_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_DIV_GRAD_KERNEL, (DeviceType::kCUDA),
                                 FLOAT16_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
