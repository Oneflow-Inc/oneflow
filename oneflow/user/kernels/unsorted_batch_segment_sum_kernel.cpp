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
#include "oneflow/core/kernel/batch_gather_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

Shape GetFlatShape(const ShapeView& shape, const int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

}  // namespace

template<DeviceType device_type, typename T, typename K>
class UnsortedBatchSegmentSumKernel final : public user_op::OpKernel {
 public:
  UnsortedBatchSegmentSumKernel() = default;
  ~UnsortedBatchSegmentSumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* data = ctx->Tensor4ArgNameAndIndex("data", 0);
    const user_op::Tensor* segment_ids = ctx->Tensor4ArgNameAndIndex("segment_ids", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t axis = segment_ids->shape().NumAxes() - 1;
    const Shape& flat_data_shape = GetFlatShape(data->shape(), axis);

    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0, out->shape().elem_cnt() * sizeof(T));
    BatchGatherKernelUtilImpl<device_type, T, K>::Backward(
        ctx->device_ctx(), data->dptr<T>(), segment_ids->dptr<K>(), flat_data_shape,
        out->shape().At(axis), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_UNSORTED_BATCH_SEGMENT_SUM_KERNEL(device, out_dtype, segment_ids_dtype)     \
  REGISTER_USER_KERNEL("unsorted_batch_segment_sum")                                         \
      .SetCreateFn<UnsortedBatchSegmentSumKernel<device, OF_PP_PAIR_FIRST(out_dtype),        \
                                                 OF_PP_PAIR_FIRST(segment_ids_dtype)>>()     \
      .SetIsMatchedHob(                                                                      \
          (user_op::HobDeviceTag() == device)                                                \
          & (user_op::HobDataType("segment_ids", 0) == OF_PP_PAIR_SECOND(segment_ids_dtype)) \
          & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(out_dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UNSORTED_BATCH_SEGMENT_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
