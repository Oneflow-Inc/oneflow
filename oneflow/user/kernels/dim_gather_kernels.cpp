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
#ifndef ONEFLOW_USER_KERNELS_DIM_GATHER_KERNELS_H_
#define ONEFLOW_USER_KERNELS_DIM_GATHER_KERNELS_H_
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
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int64_t dim = ctx->Attr<int64_t>("dim");

    if (index_tensor->shape().elem_cnt() == 0) { return; }

    const IN_T* input = input_tensor->dptr<IN_T>();
    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();

    IDX_T shape_buffer[kDimGatherMaxDimCount] = {0};
    int ndim = input_tensor->shape().NumAxes();
    ConvertShape2Array(input_tensor->shape(), shape_buffer, ndim);
    DimOpIndexNdHelper<IDX_T> input_nd_helper(shape_buffer, ndim);

    ConvertShape2Array(index_tensor->shape(), shape_buffer, ndim);
    DimOpIndexNdHelper<IDX_T> index_nd_helpr(shape_buffer, ndim);

    DimGatherFunctor<device_type, IN_T, IDX_T>()(input_nd_helper, index_nd_helpr, ndim,
                                                 index_tensor->shape().elem_cnt(), dim, index,
                                                 input, output, ctx->device_ctx());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename IN_T, typename IDX_T>
class ScatterDimKernel final : public user_op::OpKernel {
 public:
  ScatterDimKernel() = default;
  ~ScatterDimKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int64_t dim = ctx->Attr<int64_t>("dim");

    if (index_tensor->shape().elem_cnt() == 0) { return; }

    const IN_T* src = input_tensor->dptr<IN_T>();
    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();
    size_t out_bytes_size =
        out_tensor->shape().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());
    Memset<device_type>(ctx->device_ctx(), output, 0, out_bytes_size);

    IDX_T shape_buffer[kDimGatherMaxDimCount] = {0};
    int ndim = input_tensor->shape().NumAxes();
    ConvertShape2Array(input_tensor->shape(), shape_buffer, ndim);
    DimOpIndexNdHelper<IDX_T> input_nd_helper(shape_buffer, ndim);

    ConvertShape2Array(out_tensor->shape(), shape_buffer, ndim);
    DimOpIndexNdHelper<IDX_T> output_nd_helpr(shape_buffer, ndim);

    DimScatterAddFunctor<device_type, IN_T, IDX_T>()(input_nd_helper, output_nd_helpr, ndim,
                                                     input_tensor->shape().elem_cnt(), dim, index,
                                                     src, output, ctx->device_ctx());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIM_GATHER_KERNEL(device, dtype, itype)                               \
  REGISTER_USER_KERNEL("dim_gather")                                                             \
      .SetCreateFn<                                                                              \
          DimGatherKernel<device, dtype, itype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

#define REGISTER_DIM_SCATTER_KERNEL(device, dtype, itype)                               \
  REGISTER_USER_KERNEL("dim_scatter_add_like")                                                    \
      .SetCreateFn<                                                                               \
          ScatterDimKernel<device, dtype, itype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                        \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)      \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

#define REGISTER_DIM_GATHER_KERNELS_WITH_DEVICE(device) \
REGISTER_DIM_GATHER_KERNEL(device, float16, int32_t) \
REGISTER_DIM_GATHER_KERNEL(device, float, int32_t) \
REGISTER_DIM_GATHER_KERNEL(device, double, int32_t) \
REGISTER_DIM_GATHER_KERNEL(device, int32_t, int32_t) \
REGISTER_DIM_GATHER_KERNEL(device, float16, int64_t) \
REGISTER_DIM_GATHER_KERNEL(device, float, int64_t) \
REGISTER_DIM_GATHER_KERNEL(device, double, int64_t) \
REGISTER_DIM_GATHER_KERNEL(device, int32_t, int64_t) 

#define REGISTER_DIM_GATHER_KERNELS_WITH_DEVICE_INDEX64(device) \


#define REGISTER_DIM_SCATTER_ADD_LIKE_KERNELS_WITH_DEVICE(device) \
REGISTER_DIM_SCATTER_KERNEL(device, float16, int32_t) \
REGISTER_DIM_SCATTER_KERNEL(device, float, int32_t) \
REGISTER_DIM_SCATTER_KERNEL(device, double, int32_t) \
REGISTER_DIM_SCATTER_KERNEL(device, int32_t, int32_t) \
REGISTER_DIM_SCATTER_KERNEL(device, float16, int64_t) \
REGISTER_DIM_SCATTER_KERNEL(device, float, int64_t) \
REGISTER_DIM_SCATTER_KERNEL(device, double, int64_t) \
REGISTER_DIM_SCATTER_KERNEL(device, int32_t, int64_t)
 

REGISTER_DIM_GATHER_KERNELS_WITH_DEVICE(DeviceType::kCPU);
REGISTER_DIM_GATHER_KERNELS_WITH_DEVICE(DeviceType::kGPU);
REGISTER_DIM_SCATTER_ADD_LIKE_KERNELS_WITH_DEVICE(DeviceType::kCPU);
REGISTER_DIM_SCATTER_ADD_LIKE_KERNELS_WITH_DEVICE(DeviceType::kGPU);


}  // namespace user_op
}  // namespace oneflow
#endif
