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
#include "oneflow/user/kernels/batch_gather_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/gather.h"

namespace oneflow {

namespace user_op {

template<typename Context>
std::unique_ptr<ep::primitive::Gather> NewBatchGatherPrimitive(Context* ctx) {
  DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  DataType indice_type = ctx->TensorDesc4ArgNameAndIndex("indices", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::GatherFactory>(ctx->device_type(), data_type,
                                                                   indice_type);
}

template<DeviceType device_type, typename T, typename K>
class BatchGatherKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  BatchGatherKernel() = default;
  ~BatchGatherKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t axis = indices->shape_view().NumAxes() - 1;

    const Shape flatted_shape =
        Shape({in->shape().Count(0, axis), in->shape().At(axis), in->shape().Count(axis + 1)});
    const auto primitive = NewBatchGatherPrimitive(ctx);
    primitive->Launch(ctx->stream(), /*batch_size=*/flatted_shape.At(0), /*outer_size*/ 1,
                      flatted_shape.At(1), flatted_shape.At(2), in->dptr<T>(),
                      indices->shape().elem_cnt(), indices->dptr<K>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BATCH_GATHER_KERNEL(device, out_dtype, indices_dtype)        \
  REGISTER_USER_KERNEL("batch_gather")                                        \
      .SetCreateFn<BatchGatherKernel<device, OF_PP_PAIR_FIRST(out_dtype),     \
                                     OF_PP_PAIR_FIRST(indices_dtype)>>()      \
      .SetIsMatchedHob(                                                       \
          (user_op::HobDeviceType() == device)                                \
          && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_dtype)) \
          && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BATCH_GATHER_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
