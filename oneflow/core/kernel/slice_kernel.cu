#include "oneflow/core/kernel/slice_kernel.h"

namespace oneflow {

namespace {}  // namespace

template<typename T>
struct SliceKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob) {
    TODO();
  }

  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    TODO();
  }
};

#define INSTANTIATE_SLICE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SliceKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SLICE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
