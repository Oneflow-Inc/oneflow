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
#include "oneflow/user/kernels/dim_scatter_scalar_kernel_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename IN_T, typename IDX_T, template<typename T> class Opt>
class DimScatterScalarKernel final : public user_op::OpKernel {
 public:
  DimScatterScalarKernel() = default;
  ~DimScatterScalarKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");

    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();
    size_t out_bytes_size =
        out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());

    Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    const IN_T src_scalar = static_cast<IN_T>(ctx->Attr<float>("src_scalar"));

    if (input_tensor) {
      Memcpy<device_type>(ctx->stream(), output, input_tensor->dptr<IN_T>(), out_bytes_size);
    } else if (like_tensor) {
      Memset<device_type>(ctx->stream(), output, 0, out_bytes_size);
    } else {
      UNIMPLEMENTED() << "Input tensor and like tensor cannot be empty simultaneously.";
    }

    const int ndim = out_tensor->shape_view().NumAxes();
    small_vector<IDX_T, kDimGatherMaxDimCount> shape_vec(ndim);
    auto shape2dims = [&shape_vec, &ndim](const ShapeView& tensor_shape) -> void {
      std::transform(tensor_shape.ptr(), tensor_shape.ptr() + ndim, shape_vec.begin(),
                     [](int32_t dim) -> IDX_T { return static_cast<IDX_T>(dim); });
    };
    shape2dims(index_tensor->shape_view());
    DimOpIndexNdHelper<IDX_T> idx_nd_helper(shape_vec.data(), ndim);
    shape2dims(out_tensor->shape_view());
    DimOpIndexNdHelper<IDX_T> output_nd_helper(shape_vec.data(), ndim);

    int64_t upper_bound = 0;
    if (input_tensor) {
      upper_bound =
          input_tensor->shape_view().At(dim);  // ensure the idx is smaller than upperbound
    } else {
      upper_bound = like_tensor->shape_view().At(dim);  // ensure the idx is smaller than upperbound
    }

    DimScatterScalarFunctor<device_type, IN_T, IDX_T, Opt>()(
        ctx->stream(), idx_nd_helper, output_nd_helper, ndim, index_tensor->shape_view().elem_cnt(),
        dim, upper_bound, index, src_scalar, output);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCATTERSCALAR_KERNEL(op_type_name, device, dtype_pair, itype_pair, opt)      \
  REGISTER_USER_KERNEL(#op_type_name)                                                         \
      .SetCreateFn<DimScatterScalarKernel<device, OF_PP_PAIR_FIRST(dtype_pair),               \
                                          OF_PP_PAIR_FIRST(itype_pair), opt>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                   \
                       && (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(dtype_pair)) \
                       && (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(itype_pair)));

#define REGISTER_SCATTER_SCALAR_CPU_KERNELS(dtype_pair, itype_pair)                               \
  REGISTER_SCATTERSCALAR_KERNEL(dim_scatter_update_scalar, DeviceType::kCPU, dtype_pair,          \
                                itype_pair, UpdateScalarFunctor);                                 \
  REGISTER_SCATTERSCALAR_KERNEL(dim_scatter_add_scalar, DeviceType::kCPU, dtype_pair, itype_pair, \
                                AddScalarFunctor);

#define REGISTER_SCATTER_SCALAR_CUDA_KERNELS(dtype_pair, itype_pair)                               \
  REGISTER_SCATTERSCALAR_KERNEL(dim_scatter_update_scalar, DeviceType::kCUDA, dtype_pair,          \
                                itype_pair, UpdateScalarFunctor);                                  \
  REGISTER_SCATTERSCALAR_KERNEL(dim_scatter_add_scalar, DeviceType::kCUDA, dtype_pair, itype_pair, \
                                AddScalarFunctor);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_SCATTER_SCALAR_CPU_KERNELS,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_SCATTER_SCALAR_CUDA_KERNELS,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
