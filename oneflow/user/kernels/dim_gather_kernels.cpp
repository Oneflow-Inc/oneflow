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

#include "oneflow/user/kernels/dim_gather_kernel_util.h"

namespace oneflow {
namespace user_op {

#define IMPLEMENT_DIMGATHER_KERNEL_CLASS(binop)                                                 \
  template<DeviceType device_type, typename IN_T, typename IDX_T>                               \
  class DimGather##binop##Kernel final : public DimGatherBaseKernel<device_type, IN_T, IDX_T> { \
   public:                                                                                      \
    DimGather##binop##Kernel() = default;                                                       \
    ~DimGather##binop##Kernel() override = default;                                             \
                                                                                                \
   private:                                                                                     \
    void BinaryOp(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,             \
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt, \
                  int32_t dim, const IDX_T* index, const IN_T* input,                           \
                  IN_T* output) const override {                                                \
      DimGather##binop##Functor<device_type, IN_T, IDX_T>()(                                    \
          ctx, input_nd_helper, index_nd_helper, ndim, elem_cnt, dim, index, input, output);    \
    }                                                                                           \
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }                    \
  };

#define REGISTER_DIM_GATHER_KERNEL(device, dtype, itype, optypename, binop)     \
  REGISTER_USER_KERNEL(optypename)                                                       \
      .SetCreateFn<DimGather##binop##Kernel<device, dtype, itype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

#define REGISTER_DIM_GATHER_BINOP_KERNELS_DEVICE(device, optypename, binop)    \
  REGISTER_DIM_GATHER_KERNEL(device, float, int32_t, optypename, binop)   \
  REGISTER_DIM_GATHER_KERNEL(device, double, int32_t, optypename, binop)  \
  REGISTER_DIM_GATHER_KERNEL(device, int32_t, int32_t, optypename, binop) \
  REGISTER_DIM_GATHER_KERNEL(device, float, int64_t, optypename, binop)   \
  REGISTER_DIM_GATHER_KERNEL(device, double, int64_t, optypename, binop)  \
  REGISTER_DIM_GATHER_KERNEL(device, int32_t, int64_t, optypename, binop)

#define REGISTER_DIM_GATHER_CPUKERNELS(optypename, binop) \
  REGISTER_DIM_GATHER_BINOP_KERNELS_DEVICE(DeviceType::kCPU, optypename, binop);

#ifdef WITH_CUDA
#define REGISTER_DIM_GATHER_GPUKERNELS(optypename, binop)                            \
  REGISTER_DIM_GATHER_BINOP_KERNELS_DEVICE(DeviceType::kGPU, optypename, binop);          \
  REGISTER_DIM_GATHER_KERNEL(DeviceType::kGPU, float16, int32_t, optypename, binop); \
  REGISTER_DIM_GATHER_KERNEL(DeviceType::kGPU, float16, int64_t, optypename, binop);
#else
#define REGISTER_DIM_GATHER_GPUKERNELS(optypename, binop)
#endif  // WITH_CUDA

#define REGISTER_GATHER_KERNEL(optypename, binop)    \
  REGISTER_DIM_GATHER_CPUKERNELS(optypename, binop); \
  REGISTER_DIM_GATHER_GPUKERNELS(optypename, binop);

template<DeviceType device_type, typename IN_T, typename IDX_T>
class DimGatherBaseKernel : public user_op::OpKernel {
 public:
  DimGatherBaseKernel() = default;
  ~DimGatherBaseKernel() override = default;
  virtual void BinaryOp(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                        const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim,
                        int64_t elem_cnt, int32_t dim, const IDX_T* index, const IN_T* input,
                        IN_T* output) const = 0;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");

    const IN_T* input = input_tensor->dptr<IN_T>();
    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();

    int ndim = input_tensor->shape().NumAxes();
    fixed_vector<IDX_T, kDimGatherMaxDimCount> shape_vec(ndim);
    auto shape2dims = [&shape_vec, &ndim](const ShapeView& tensor_shape) -> void {
      std::transform(tensor_shape.ptr(), tensor_shape.ptr() + ndim, shape_vec.begin(),
                     [](int64_t dim) -> IDX_T { return static_cast<IDX_T>(dim); });
    };
    shape2dims(input_tensor->shape());
    DimOpIndexNdHelper<IDX_T> input_nd_helper(shape_vec.data(), ndim);
    shape2dims(index_tensor->shape());
    DimOpIndexNdHelper<IDX_T> index_nd_helper(shape_vec.data(), ndim);

    BinaryOp(ctx->device_ctx(), input_nd_helper, index_nd_helper, ndim,
             index_tensor->shape().elem_cnt(), dim, index, input, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

IMPLEMENT_DIMGATHER_KERNEL_CLASS(Update);
REGISTER_GATHER_KERNEL("dim_gather", Update);

}  // namespace user_op
}  // namespace oneflow
