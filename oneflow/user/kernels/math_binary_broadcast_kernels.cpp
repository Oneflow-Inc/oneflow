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
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/ops/math_binary_broadcast_seq.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {


bool isContiguous(const int64_t* dim_vec, const int64_t* stride_vec) {
  // TODO:
  return false;
}

template<DeviceType device_type, typename T, typename K, ep::primitive::BinaryOp op>
class MathBinaryBroadcastKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MathBinaryBroadcastKernel() = default;
  ~MathBinaryBroadcastKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* z = ctx->Tensor4ArgNameAndIndex("z", 0);

    DimVector x_dim_vec;
    DimVector y_dim_vec;
    DimVector z_dim_vec;

    x->shape().ToDimVector(&x_dim_vec);
    y->shape().ToDimVector(&y_dim_vec);
    z->shape().ToDimVector(&z_dim_vec);

    // TODO: hasAttr? test if strides attributes exist, if not exists, use normal premitive
    // add Stride as a valid Attr type

    const auto& x_stride_vec = ctx->Attr<std::vector<int64_t>>("x_stride");
    const auto& y_stride_vec = ctx->Attr<std::vector<int64_t>>("y_stride");
    const auto& z_stride_vec = ctx->Attr<std::vector<int64_t>>("z_stride");


    auto primitive = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), y->data_type(), x->shape().NumAxes() > y->shape().NumAxes() ? x->shape().NumAxes() : y->shape().NumAxes());
    CHECK(primitive);


    if(x_stride_vec.size() == 0 && y_stride_vec.size() == 0) {
      primitive->Launch(ctx->stream(), 
        x->shape().NumAxes(), x_dim_vec.data(), x->dptr<T>(),
        y->shape().NumAxes(), y_dim_vec.data(), y->dptr<T>(),
        z->mut_dptr<K>());
    } else {
      // TODO: if contiguous, use old primitive api
      // TODO: if z_strides is not given (not inpliace), generate z_strides as contiguous (maybe put into primitive)
      if(z_stride_vec.size() == 0) {
        Shape target_shape;
        z->shape().ToShape(&target_shape);
        Stride target_stride(target_shape);

        LOG(INFO) << "Calling primitive (contiguous, non-inplace)";
        primitive->Launch(ctx->stream(), 
          x->shape().NumAxes(), x_dim_vec.data(), x_stride_vec.data(), x->dptr<T>(),
          y->shape().NumAxes(), y_dim_vec.data(), y_stride_vec.data(), y->dptr<T>(),
          z->shape().NumAxes(), z_dim_vec.data(), target_stride.StrideVec().data(), z->mut_dptr<K>());
      } else {
        primitive->Launch(ctx->stream(), 
          x->shape().NumAxes(), x_dim_vec.data(), x_stride_vec.data(), x->dptr<T>(),
          y->shape().NumAxes(), y_dim_vec.data(), y_stride_vec.data(), y->dptr<T>(),
          z->shape().NumAxes(), z_dim_vec.data(), z_stride_vec.data(), z->mut_dptr<K>());
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MATH_BINARY_BROADCAST_KERNEL(math_type_pair, device, data_type_pair) \
  REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                              \
      .SetCreateFn<MathBinaryBroadcastKernel<                                         \
          device, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(data_type_pair), \
          OF_PP_PAIR_SECOND(math_type_pair)>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                           \
                       && (user_op::HobDataType("z", 0) == OF_PP_PAIR_SECOND(data_type_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
                                 MATH_BINARY_BROADCAST_PRIMITIVE_OP_SEQ, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)
// // gpu half
// #ifdef WITH_CUDA
// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_KERNEL,
//                                  MATH_BINARY_BROADCAST_FUNC_SEQ, (DeviceType::kCUDA),
//                                  FLOAT16_DATA_TYPE_SEQ)
// #endif

// #define REGISTER_MATH_BINARY_BROADCAST_LOGICAL_KERNEL(math_type_pair, device, data_type_pair) \
//   REGISTER_USER_KERNEL(OF_PP_PAIR_FIRST(math_type_pair))                                      \
//       .SetCreateFn<MathBinaryBroadcastKernel<                                                 \
//           device, OF_PP_PAIR_FIRST(data_type_pair), int8_t,                                   \
//           &NdarrayUtil<device, OF_PP_PAIR_FIRST(data_type_pair)>::OF_PP_CAT(                  \
//               Broadcast, OF_PP_PAIR_SECOND(math_type_pair))>>()                               \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                                   \
//                        && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(data_type_pair)) \
//                        && (user_op::HobDataType("z", 0) == DataType::kInt8));

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MATH_BINARY_BROADCAST_LOGICAL_KERNEL,
//                                  MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ, DEVICE_TYPE_SEQ,
//                                  ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ)

}  // namespace oneflow
