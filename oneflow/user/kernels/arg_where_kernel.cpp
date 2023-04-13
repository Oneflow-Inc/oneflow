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
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/user/kernels/arg_where_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename IN_T, typename OUT_T>
class ArgWhereKernel final : public user_op::OpKernel {
 public:
  ArgWhereKernel() = default;
  ~ArgWhereKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    int64_t ndims = ctx->Tensor4ArgNameAndIndex("input", 0)->shape_view().NumAxes();
    if (ndims == 0) {
      // 0-dim tensor, elem_cnt of input is 1
      CHECK_EQ(ctx->Tensor4ArgNameAndIndex("input", 0)->shape_view().elem_cnt(), 1);
      SetOutputSize<device_type, IN_T, OUT_T>(
          ctx->stream(), ctx->Tensor4ArgNameAndIndex("input", 0)->dptr<IN_T>(),
          ctx->Tensor4ArgNameAndIndex("output_size", 0)->mut_dptr<OUT_T>());
      return;
    }
    SwitchNdimCompute(SwitchCase(ndims), ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

#define COMPUTE_SWITCH_ENTRY(func_name, ndim) func_name<ndim>
  DEFINE_STATIC_SWITCH_FUNC(void, NdimCompute, COMPUTE_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef COMPUTE_SWITCH_ENTRY

  template<int NDIM>
  static void NdimCompute(user_op::KernelComputeContext* ctx) {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* output_size = ctx->Tensor4ArgNameAndIndex("output_size", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
    size_t tmp_size = tmp ? tmp->shape_view().elem_cnt() * GetSizeOfDataType(tmp->data_type()) : 0;
    ArgWhereKernelUtil<device_type, IN_T, OUT_T, NDIM>::ArgWhere(
        ctx->stream(), input->shape_view(), input->dptr<IN_T>(), tmp_ptr, tmp_size,
        output->mut_dptr<OUT_T>(), output_size->mut_dptr<OUT_T>());
  }
};

template<DeviceType device_type, typename IN_T, typename OUT_T, int NDIM>
size_t GetWorkspaceBytesSize(int64_t elem_cnt) {
  return ArgWhereKernelUtil<device_type, IN_T, OUT_T, NDIM>::GetWorkspaceBytesSize(nullptr,
                                                                                   elem_cnt);
}

template<DeviceType device_type>
struct SwitchUtil;

template<>
struct SwitchUtil<DeviceType::kCPU> {
#define SWITCH_ENTRY(func_name, device, itype, otype, ndim) func_name<device, itype, otype, ndim>

  DEFINE_STATIC_SWITCH_FUNC(
      size_t, GetWorkspaceBytesSize, SWITCH_ENTRY,
      MAKE_DEVICE_TYPE_CTRV_SEQ(OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU)),
      MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ
                                  FLOAT16_DATA_TYPE_SEQ),
      MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ), MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef SWITCH_ENTRY
};

#ifdef WITH_CUDA

template<>
struct SwitchUtil<DeviceType::kCUDA> {
#define SWITCH_ENTRY(func_name, device, itype, otype, ndim) func_name<device, itype, otype, ndim>

  DEFINE_STATIC_SWITCH_FUNC(
      size_t, GetWorkspaceBytesSize, SWITCH_ENTRY,
      MAKE_DEVICE_TYPE_CTRV_SEQ(OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA)),
      MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ
                                  HALF_DATA_TYPE_SEQ),
      MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ), MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef SWITCH_ENTRY
};

#endif  // WITH_CUDA

template<DeviceType device_type>
size_t InferTempStorageBytesSize(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  if (input_shape.NumAxes() == 0) { return 0; }
  DataType input_dtype = ctx->InputDType("input", 0);
  DataType output_dtype = ctx->OutputDType("output", 0);
  return SwitchUtil<device_type>::SwitchGetWorkspaceBytesSize(
      SwitchCase(device_type, input_dtype, output_dtype, input_shape.NumAxes()),
      input_shape.elem_cnt());
}

}  // namespace

#define REGISTER_ARG_WHERE_KERNEL(device, itype, otype)                                          \
  REGISTER_USER_KERNEL("argwhere")                                                               \
      .SetCreateFn<ArgWhereKernel<device, itype, otype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<itype>::value)        \
                       && (user_op::HobDataType("output", 0) == GetDataType<otype>::value)       \
                       && (user_op::HobDataType("output_size", 0) == GetDataType<otype>::value)) \
      .SetInferTmpSizeFn(InferTempStorageBytesSize<device>);

#define REGISTER_ARG_WHERE_KERNEL_WITH_DTYPE_PAIR(device, itype_pair, otype_pair) \
  REGISTER_ARG_WHERE_KERNEL(device, OF_PP_PAIR_FIRST(itype_pair), OF_PP_PAIR_FIRST(otype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    REGISTER_ARG_WHERE_KERNEL_WITH_DTYPE_PAIR, DEVICE_TYPE_SEQ,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ARG_WHERE_KERNEL_WITH_DTYPE_PAIR, (DeviceType::kCPU),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ARG_WHERE_KERNEL_WITH_DTYPE_PAIR, (DeviceType::kCUDA),
                                 HALF_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif  // WITH_CUDA
}  // namespace oneflow
