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

template<DeviceType device_type, typename IN_T, typename OUT_T, int NDIM>
class ArgWhereKernel final : public user_op::OpKernel {
 public:
  ArgWhereKernel() = default;
  ~ArgWhereKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* output_size = ctx->Tensor4ArgNameAndIndex("output_size", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
    size_t tmp_size = tmp ? tmp->shape().elem_cnt() * GetSizeOfDataType(tmp->data_type()) : 0;
    ArgWhereKernelUtil<device_type, IN_T, OUT_T, NDIM>::ArgWhere(
        ctx->device_ctx(), input->shape(), input->dptr<IN_T>(), tmp_ptr, tmp_size,
        output->mut_dptr<OUT_T>(), output_size->mut_dptr<OUT_T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename IN_T, typename OUT_T, int NDIM>
size_t GetWorkspaceBytesSize(int64_t elem_cnt) {
  return ArgWhereKernelUtil<device_type, IN_T, OUT_T, NDIM>::GetWorkspaceBytesSize(nullptr,
                                                                                   elem_cnt);
}

struct SwitchUtil {
#define SWITCH_ENTRY(func_name, device, itype, otype, ndim) func_name<device, itype, otype, ndim>

  DEFINE_STATIC_SWITCH_FUNC(size_t, GetWorkspaceBytesSize, SWITCH_ENTRY,
                            MAKE_DEVICE_TYPE_CTRV_SEQ(DEVICE_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ),
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef SWITCH_ENTRY
};

size_t InferTempStorageBytesSize(user_op::InferContext* ctx) {
  const std::string& device_tag = ctx->user_op_conf().op_conf().device_tag();
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(device_tag));
  const Shape* input_shape = ctx->Shape4ArgNameAndIndex("input", 0);
  DataType input_dtype = *ctx->Dtype4ArgNameAndIndex("input", 0);
  DataType output_dtype = *ctx->Dtype4ArgNameAndIndex("output", 0);
  return SwitchUtil::SwitchGetWorkspaceBytesSize(
      SwitchCase(device_type, input_dtype, output_dtype, input_shape->NumAxes()),
      input_shape->elem_cnt());
}

}  // namespace

#define REGISTER_ARG_WHERE_KERNEL(device, itype, otype, ndim)                                  \
  REGISTER_USER_KERNEL("argwhere")                                                             \
      .SetCreateFn<ArgWhereKernel<device, itype, otype, ndim>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                     \
                       & (user_op::HobDataType("input", 0) == GetDataType<itype>::value)       \
                       & (user_op::HobDataType("output", 0) == GetDataType<otype>::value)      \
                       & (user_op::HobDataType("output_size", 0) == GetDataType<otype>::value) \
                       & (user_op::HobNumAxes("input", 0) == ndim))                            \
      .SetInferTmpSizeFn(InferTempStorageBytesSize);

#define REGISTER_ARG_WHERE_KERNEL_WITH_DTYPE_PAIR(device, itype_pair, otype_pair, ndim)         \
  REGISTER_ARG_WHERE_KERNEL(device, OF_PP_PAIR_FIRST(itype_pair), OF_PP_PAIR_FIRST(otype_pair), \
                            ndim)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ARG_WHERE_KERNEL_WITH_DTYPE_PAIR, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, DIM_SEQ)

}  // namespace oneflow
