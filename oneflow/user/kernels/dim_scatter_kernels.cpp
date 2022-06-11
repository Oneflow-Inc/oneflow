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
        out_tensor->shape().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());

    Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    const IN_T* src = src_tensor->dptr<IN_T>();

    if (input_tensor) {
      Memcpy<device_type>(ctx->stream(), output, input_tensor->dptr<IN_T>(), out_bytes_size);
    } else if (like_tensor) {
      Memset<device_type>(ctx->stream(), output, 0, out_bytes_size);
    } else {
      UNIMPLEMENTED() << "Input tensor and like tensor cannot be empty simultaneously.";
    }

    const int ndim = src_tensor->shape().NumAxes();
    small_vector<IDX_T, kDimGatherMaxDimCount> shape_vec(ndim);
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

    DimScatterFunctor<device_type, IN_T, IDX_T, Opt>()(
        ctx->stream(), src_nd_helper, idx_nd_helper, output_nd_helper, ndim,
        index_tensor->shape().elem_cnt(), dim, upper_bound, index, src, output);
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
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, int32_t, int32_t, opt); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, bool, int64_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, float, int64_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, double, int64_t, opt);  \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCPU, int32_t, int64_t, opt);

#define REGISTER_DIM_SCATTER_LIKE_CUDA_KERNELS(op_type, opt)                           \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, bool, int32_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, float, int32_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, double, int32_t, opt);  \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, int32_t, int32_t, opt); \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, bool, int64_t, opt);    \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, float, int64_t, opt);   \
  REGISTER_DIM_SCATTER_LIKE_KERNEL(op_type, DeviceType::kCUDA, double, int64_t, opt);  \
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
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_update, DeviceType::kCPU, dtype_pair, itype_pair, \
                              BinOpUpdateFunctor);

#define REGISTER_DIM_SCATTER_CUDA_KERNELS(dtype_pair, itype_pair)                            \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_add, DeviceType::kCUDA, dtype_pair, itype_pair,    \
                              BinOpAddFunctor);                                              \
  REGISTER_DIM_SCATTER_KERNEL(dim_scatter_update, DeviceType::kCUDA, dtype_pair, itype_pair, \
                              BinOpUpdateFunctor);

REGISTER_DIM_SCATTER_LIKE_CPU_KERNELS("dim_scatter_add_like", BinOpAddFunctor);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_DIM_SCATTER_CPU_KERNELS,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
REGISTER_DIM_SCATTER_LIKE_CUDA_KERNELS("dim_scatter_add_like", BinOpAddFunctor);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_DIM_SCATTER_CUDA_KERNELS,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
