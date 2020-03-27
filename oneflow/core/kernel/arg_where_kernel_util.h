#ifndef ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<typename T, typename I>
size_t InferSelectTrueTmpBufferSize(cudaStream_t stream, int num_items);

#define INSTANTIATE_INFER_SELECT_TRUE_TMP_BUFFER_SIZE(dtype_pair, itype_pair) \
  template size_t InferSelectTrueTmpBufferSize<OF_PP_PAIR_FIRST(dtype_pair),  \
                                               OF_PP_PAIR_FIRST(itype_pair)>(cudaStream_t, int);

#define REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, ndims)         \
  NEW_REGISTER_KERNEL(OperatorConf::kArgWhereConf, kernel<dtype, itype, ndims>)       \
      .SetIsMatchedPred([](const KernelConf& conf) {                                  \
        return (device_type_v == conf.op_attribute().op_conf().device_type())         \
               && (GetDataType<itype>::value == conf.data_type())                     \
               && (GetDataType<dtype>::value == conf.arg_where_conf().in_data_type()) \
               && (ndims == conf.arg_where_conf().num_axes());                        \
      });

#define REGISTER_ARG_WHERE_KERNELS_AT_NDIMS(kernel, device_type_v, dtype, itype) \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 1);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 2);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 3);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 4);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 5);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 6);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 7);             \
  REGISTER_ARG_WHERE_KERNEL(kernel, device_type_v, dtype, itype, 8);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_
