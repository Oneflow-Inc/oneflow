#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastBinaryKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastBinaryKernel);
  BroadcastBinaryKernel() = default;
  virtual ~BroadcastBinaryKernel() = default;
};

template<DeviceType device_type, typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct BroadcastBinaryKernelHelper final {
  static void Forward(DeviceCtx* ctx, XpuVarNdarray<T>&& y, const XpuVarNdarray<const T>& a,
                      const XpuVarNdarray<const T>& b);
};

template<DeviceType device_type, typename T, const T (*binary_func)(const T, const T)>
struct BroadcastBinaryKernelUtil final {
  static void Forward(const KernelCtx& kernel_ctx,
                      const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* out_blob = BnInOp2Blob("out");
    switch (out_blob->shape().NumAxes()) {
#define MAKE_ENTRY(NDIMS)                                                                  \
  case NDIMS:                                                                              \
    return BroadcastBinaryKernelHelper<device_type, T, NDIMS, binary_func>::Forward(       \
        kernel_ctx.device_ctx, XpuVarNdarray<T>(out_blob), XpuVarNdarray<const T>(a_blob), \
        XpuVarNdarray<const T>(b_blob));

      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DIM_SEQ)
#undef MAKE_ENTRY
      default: UNIMPLEMENTED();
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
