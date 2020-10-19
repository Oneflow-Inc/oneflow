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
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/arg_where_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename I, size_t NDims>
class ArgWhereKernel : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgWhereKernel);
  ArgWhereKernel() = default;
  ~ArgWhereKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    if (in->shape().elem_cnt() == 0) { return; }
    Blob* out = BnInOp2Blob("out");
    Blob* out_size = BnInOp2Blob("out_size");
    Blob* tmp = BnInOp2Blob("tmp");
    ArgWhereKernelUtil<device_type, T, I, NDims>::ArgWhere(
        ctx.device_ctx, in->shape(), in->dptr<T>(), tmp ? tmp->mut_dptr() : nullptr,
        tmp ? tmp->ByteSizeOfBlobBody() : 0, out->mut_dptr<I>(), out_size->mut_dptr<I>());
  }
};

#define REGISTER_ARG_WHERE_KERNEL(device_type_v, dtype, itype, ndims)                  \
  NEW_REGISTER_KERNEL(OperatorConf::kArgWhereConf,                                     \
                      ArgWhereKernel<device_type_v, dtype, itype, ndims>)              \
      .SetIsMatchedPred([](const KernelConf& conf) -> bool {                           \
        return (conf.op_attribute().op_conf().device_tag() == ToString(device_type_v)) \
               && (GetDataType<itype>::value == conf.data_type())                      \
               && (GetDataType<dtype>::value == conf.arg_where_conf().in_data_type())  \
               && (ndims == conf.arg_where_conf().num_axes());                         \
      });

#define REGISTER_ARG_WHERE_KERNELS_AT_NDIMS(device_type_v, dtype, itype) \
  REGISTER_ARG_WHERE_KERNEL(device_type_v, dtype, itype, 1)              \
  REGISTER_ARG_WHERE_KERNEL(device_type_v, dtype, itype, 2)              \
  REGISTER_ARG_WHERE_KERNEL(device_type_v, dtype, itype, 3)              \
  REGISTER_ARG_WHERE_KERNEL(device_type_v, dtype, itype, 4)              \
  REGISTER_ARG_WHERE_KERNEL(device_type_v, dtype, itype, 5)

#define REGISTER_ARG_WHERE_KERNELS(device_type_v, dtype_pair, itype_pair)          \
  REGISTER_ARG_WHERE_KERNELS_AT_NDIMS(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                      OF_PP_PAIR_FIRST(itype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ARG_WHERE_KERNELS, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
