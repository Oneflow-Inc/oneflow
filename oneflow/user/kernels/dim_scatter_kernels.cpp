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

template<DeviceType device_type, typename IN_T, typename IDX_T, template<typename T> class Opt>
class DimScatterKernel final : public user_op::OpKernel {
 public:
  DimScatterKernel() = default;
  ~DimScatterKernel() override = default;

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
        out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());

    Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    const IN_T* src = src_tensor->dptr<IN_T>();

    if (input_tensor) {
      Memcpy<device_type>(ctx->stream(), output, input_tensor->dptr<IN_T>(), out_bytes_size);
    } else if (like_tensor) {
      Memset<device_type>(ctx->stream(), output, 0, out_bytes_size);
    } else {
      UNIMPLEMENTED() << "Input tensor and like tensor cannot be empty simultaneously.";
    }

    const Shape src_shape = ExpandDimIf0D(src_tensor->shape_view());
    const Shape index_shape = ExpandDimIf0D(index_tensor->shape_view());
    const int ndim = src_shape.NumAxes();
    DimOpIndexNdHelper<IDX_T> src_nd_helper(src_shape.data(), ndim);
    DimOpIndexNdHelper<IDX_T> idx_nd_helper(index_shape.data(), ndim);
    DimOpIndexNdHelper<IDX_T> output_nd_helper(out_tensor->shape_view().data(), ndim);

    const int64_t upper_bound = [&]() {
      if (input_tensor) {
        const Shape input_shape = ExpandDimIf0D(input_tensor->shape_view());
        return input_shape.At(dim);
      } else {
        const Shape like_shape = ExpandDimIf0D(like_tensor->shape_view());
        return like_shape.At(dim);
      }
    }();

    DimScatterFunctor<device_type, IN_T, IDX_T, Opt>()(
        ctx->stream(), src_nd_helper, idx_nd_helper, output_nd_helper, ndim, index_shape.elem_cnt(),
        dim, upper_bound, index, src, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, device, dtype, itype, opt)             \
  REGISTER_USER_KERNEL(op_type)                                                          \
      .SetCreateFn<DimScatterKernel<device, dtype, itype, opt>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                              \
                       && (user_op::HobDataType("like", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("index", 0) == GetDataType<itype>::value));

#define REGISTER_DIM_SCATTER_LIKE_CPU_KERNELS(op_type, opt)                           \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, bool, int32_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, float, int32_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, double, int32_t, opt);  \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, float16, int32_t, opt); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, int32_t, int32_t, opt); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, bool, int64_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, float, int64_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, double, int64_t, opt);  \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, float16, int64_t, opt); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, int32_t, int64_t, opt);

#define REGISTER_DIM_SCATTER_LIKE_CUDA_KERNELS(op_type, opt)                           \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, bool, int32_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, float, int32_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, double, int32_t, opt);  \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, half, int32_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, int32_t, int32_t, opt); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, bool, int64_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, float, int64_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, double, int64_t, opt);  \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, half, int64_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, int32_t, int64_t, opt);

#define REGISTER_DIM_SCATTER_KERNEL(op_type, device, dtype_pair, itype_pair, opt)             \
  REGISTER_USER_KERNEL(#op_type)                                                              \
      .SetCreateFn<DimScatterKernel<device, OF_PP_PAIR_FIRST(dtype_pair),                     \
                                    OF_PP_PAIR_FIRST(itype_pair), opt>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                   \
                       && (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(dtype_pair)) \
                       && (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(itype_pair)));

#define REGISTER_DIM_SCATTER_CPU_KERNELS(dtype_pair, itype_pair)                            \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_add, DeviceType::kCPU, dtype_pair, itype_pair,    \
                              BinOpAddFunctor);                                             \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_mul, DeviceType::kCPU, dtype_pair, itype_pair,    \
                              BinOpMulFunctor);                                             \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_update, DeviceType::kCPU, dtype_pair, itype_pair, \
                              BinOpUpdateFunctor);

#define REGISTER_DIM_SCATTER_CUDA_KERNELS(dtype_pair, itype_pair)                            \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_add, DeviceType::kCUDA, dtype_pair, itype_pair,    \
                              BinOpAddFunctor);                                              \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_mul, DeviceType::kCUDA, dtype_pair, itype_pair,    \
                              BinOpMulFunctor);                                              \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_update, DeviceType::kCUDA, dtype_pair, itype_pair, \
                              BinOpUpdateFunctor);

REGISTER_DIM_SCATTER_LIKE_CPU_KERNELS("dim_scatter_add_like", BinOpAddFunctor);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DIM_SCATTER_CPU_KERNELS,
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
REGISTER_DIM_SCATTER_LIKE_CUDA_KERNELS("dim_scatter_add_like", BinOpAddFunctor);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DIM_SCATTER_CUDA_KERNELS,
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
