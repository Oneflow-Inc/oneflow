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
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/user/kernels/dim_scatter_kernel_util.h"

namespace oneflow {
namespace user_op {

#define IMPLEMENT_DIMSCATTER_KERNEL_CLASS(binop)                                                  \
  template<DeviceType device_type, typename IN_T, typename IDX_T>                                 \
  class DimScatter##binop##Kernel final : public DimScatterBaseKernel<device_type, IN_T, IDX_T> { \
   public:                                                                                        \
    DimScatter##binop##Kernel() = default;                                                        \
    ~DimScatter##binop##Kernel() override = default;                                              \
                                                                                                  \
   private:                                                                                       \
    void BinaryOp(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,                 \
                  const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,                                 \
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim, int64_t elem_cnt,  \
                  int32_t dim, int64_t upper_bound, const IDX_T* index, const IN_T* src,          \
                  IN_T* output) const override {                                                  \
      DimScatter##binop##Functor<device_type, IN_T, IDX_T>()(                                     \
          ctx, src_nd_helper, idx_nd_helper, output_nd_helper, ndim, elem_cnt, dim, upper_bound,  \
          index, src, output);                                                                    \
    }                                                                                             \
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }                      \
  }

#define REGISTER_DIM_SCATTER_LIKE_KERNEL(device, dtype, itype, optypename, binop)       \
  REGISTER_USER_KERNEL(optypename)                                                      \
      .SetCreateFn<DimScatter##binop##Kernel<device, dtype, itype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("like", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

#define REGISTER_DIM_SCATTER_BINOP_LIKE_KERNELS_DEVICE(device, optypename, binop) \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(device, float, int32_t, optypename, binop)     \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(device, double, int32_t, optypename, binop)    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(device, int32_t, int32_t, optypename, binop)   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(device, float, int64_t, optypename, binop)     \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(device, double, int64_t, optypename, binop)    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(device, int32_t, int64_t, optypename, binop)

#define REGISTER_DIM_SCATTER_LIKE_CPUKERNELS(optypename, binop) \
  REGISTER_DIM_SCATTER_BINOP_LIKE_KERNELS_DEVICE(DeviceType::kCPU, optypename, binop);

#ifdef WITH_CUDA
#define REGISTER_DIM_SCATTER_LIKE_GPUKERNELS(optypename, binop)                            \
  REGISTER_DIM_SCATTER_BINOP_LIKE_KERNELS_DEVICE(DeviceType::kGPU, optypename, binop);     \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(DeviceType::kGPU, float16, int32_t, optypename, binop); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(DeviceType::kGPU, float16, int64_t, optypename, binop);
#else
#define REGISTER_DIM_SCATTER_LIKE_GPUKERNELS(optypename, binop)
#endif  // WITH_CUDA

#define REGISTER_SCATTER_LIKE_KERNEL(optypename, binop)    \
  REGISTER_DIM_SCATTER_LIKE_CPUKERNELS(optypename, binop); \
  REGISTER_DIM_SCATTER_LIKE_GPUKERNELS(optypename, binop);

#define REGISTER_DIM_SCATTER_KERNEL(device, dtype, itype, optypename, binop)             \
  REGISTER_USER_KERNEL(optypename)                                                       \
      .SetCreateFn<DimScatter##binop##Kernel<device, dtype, itype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::value))

#define REGISTER_DIM_SCATTER_BINOP_KERNELS_DEVICE(device, optypename, binop) \
  REGISTER_DIM_SCATTER_KERNEL(device, float, int32_t, optypename, binop);    \
  REGISTER_DIM_SCATTER_KERNEL(device, double, int32_t, optypename, binop);   \
  REGISTER_DIM_SCATTER_KERNEL(device, int32_t, int32_t, optypename, binop);  \
  REGISTER_DIM_SCATTER_KERNEL(device, float, int64_t, optypename, binop);    \
  REGISTER_DIM_SCATTER_KERNEL(device, double, int64_t, optypename, binop);   \
  REGISTER_DIM_SCATTER_KERNEL(device, int32_t, int64_t, optypename, binop);

#define REGISTER_DIM_SCATTER_CPUKERNELS(optypename, binop) \
  REGISTER_DIM_SCATTER_BINOP_KERNELS_DEVICE(DeviceType::kCPU, optypename, binop);

#ifdef WITH_CUDA
#define REGISTER_DIM_SCATTER_GPUKERNELS(optypename, binop)                            \
  REGISTER_DIM_SCATTER_BINOP_KERNELS_DEVICE(DeviceType::kGPU, optypename, binop);     \
  REGISTER_DIM_SCATTER_KERNEL(DeviceType::kGPU, float16, int32_t, optypename, binop); \
  REGISTER_DIM_SCATTER_KERNEL(DeviceType::kGPU, float16, int64_t, optypename, binop);
#else
#define REGISTER_DIM_SCATTER_GPUKERNELS(optypename, binop)
#endif  // WITH_CUDA

#define REGISTER_SCATTER_KERNEL(optypename, binop)    \
  REGISTER_DIM_SCATTER_CPUKERNELS(optypename, binop); \
  REGISTER_DIM_SCATTER_GPUKERNELS(optypename, binop);

template<DeviceType device_type, typename IN_T, typename IDX_T>
class DimScatterBaseKernel : public user_op::OpKernel {
 public:
  DimScatterBaseKernel() = default;
  ~DimScatterBaseKernel() override = default;
  virtual void BinaryOp(DeviceCtx* ctx, const DimOpIndexNdHelper<IDX_T>& src_nd_helper,
                        const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                        const DimOpIndexNdHelper<IDX_T>& output_nd_helper, int ndim,
                        int64_t elem_cnt, int32_t dim, int64_t upper_bound, const IDX_T* index,
                        const IN_T* src, IN_T* output) const = 0;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const Tensor* src_tensor = ctx->Tensor4ArgNameAndIndex("src", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");

    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();
    size_t out_bytes_size =
        out_tensor->shape().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());

    Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    const IN_T* src = src_tensor->dptr<IN_T>();

    if (input_tensor) {
      Memcpy<device_type>(ctx->device_ctx(), output, input_tensor->dptr<IN_T>(), out_bytes_size);
    } else if (like_tensor) {
      Memset<device_type>(ctx->device_ctx(), output, 0, out_bytes_size);
    } else {
      std::cout << "Unimplemented Error" << std::endl;
      throw Error::Unimplemented();
    }

    const int ndim = src_tensor->shape().NumAxes();
    fixed_vector<IDX_T, kDimGatherMaxDimCount> shape_vec(ndim);
    auto shape2dims = [&shape_vec, &ndim](const ShapeView& tensor_shape) -> void {
      std::transform(tensor_shape.ptr(), tensor_shape.ptr() + ndim, shape_vec.begin(),
                     [](int32_t dim) -> IDX_T { return static_cast<IDX_T>(dim); });
    };
    shape2dims(src_tensor->shape());
    DimOpIndexNdHelper<IDX_T> src_nd_helper(shape_vec.data(), ndim);
    shape2dims(index_tensor->shape());
    DimOpIndexNdHelper<IDX_T> idx_nd_helper(shape_vec.data(), ndim);
    shape2dims(out_tensor->shape());
    DimOpIndexNdHelper<IDX_T> output_nd_helper(shape_vec.data(), ndim);

    int64_t upper_bound = 0;
    if (input_tensor) {
      upper_bound = input_tensor->shape().At(dim);  // ensure the idx is smaller than upperbound
    } else {
      upper_bound = like_tensor->shape().At(dim);  // ensure the idx is smaller than upperbound
    }

    BinaryOp(ctx->device_ctx(), src_nd_helper, idx_nd_helper, output_nd_helper, ndim,
             index_tensor->shape().elem_cnt(), dim, upper_bound, index, src, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

IMPLEMENT_DIMSCATTER_KERNEL_CLASS(Add);
IMPLEMENT_DIMSCATTER_KERNEL_CLASS(Update);

REGISTER_SCATTER_LIKE_KERNEL("dim_scatter_add_like", Add);
REGISTER_SCATTER_LIKE_KERNEL("dim_scatter_update_like", Update);
REGISTER_SCATTER_KERNEL("dim_scatter_add", Add);
REGISTER_SCATTER_KERNEL("dim_scatter_update", Update);

}  // namespace user_op
}  // namespace oneflow
