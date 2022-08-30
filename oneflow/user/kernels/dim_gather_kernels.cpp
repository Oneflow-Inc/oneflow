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

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/dim_gather_kernel_util.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename IDX_T>
void ConvertShape2Array(const ShapeView& shape_view, IDX_T* array, int64_t num_axis) {
  FOR_RANGE(int64_t, i, 0, num_axis) { array[i] = shape_view.At(i); }
}

}  // namespace

template<DeviceType device_type, typename IN_T, typename IDX_T>
class DimGatherKernel final : public user_op::OpKernel {
 public:
  DimGatherKernel() = default;
  ~DimGatherKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    if (input_tensor->shape_view().elem_cnt() == 0) { return; }
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");

    const IN_T* input = input_tensor->dptr<IN_T>();
    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();

    const Shape in_shape = ExpandDimIf0D(input_tensor->shape_view());
    const auto ndim = in_shape.NumAxes();
    const auto dim_length = in_shape.At(dim);

    DimOpIndexNdHelper<IDX_T> input_nd_helper(in_shape.data(), ndim);
    DimOpIndexNdHelper<IDX_T> index_nd_helper(index_tensor->shape_view().data(), ndim);
    DimGatherFunctor<device_type, IN_T, IDX_T>()(ctx->stream(), input_nd_helper, index_nd_helper,
                                                 ndim, index_tensor->shape_view().elem_cnt(),
                                                 dim_length, dim, index, input, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIM_GATHER_KERNEL(device, dtype_pair, itype_pair)                               \
  REGISTER_USER_KERNEL("dim_gather")                                                             \
      .SetCreateFn<                                                                              \
          DimGatherKernel<device, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(itype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(dtype_pair))    \
                       && (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(itype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_DIM_GATHER_KERNEL, (DeviceType::kCPU),
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DIM_GATHER_KERNEL, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     FLOAT16_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
