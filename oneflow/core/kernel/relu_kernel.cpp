#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  ReluKernelUtil<device_type, T>::Forward(ctx, out_blob->shape().elem_cnt(),
                                          in_blob->dptr<T>(),
                                          out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  ReluKernelUtil<device_type, T>::Backward(
      ctx, in_blob->shape().elem_cnt(), out_diff_blob->dptr<T>(),
      in_blob->dptr<T>(), in_diff_blob->mut_dptr<T>());
}

template<typename T>
class ReluKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n, const T* in,
                      T* out) {
    for (int64_t i = 0; i < n; ++i) {
      out[i] = std::max(in[i], static_cast<T>(0.0));
    }
  }

  static void Backward(const KernelCtx& ctx, const int64_t n, const T* out_diff,
                       const T* in, T* in_diff) {
    for (int64_t i = 0; i < n; ++i) {
      in_diff[i] = in[i] > 0 ? out_diff[i] : 0;
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReluConf, ReluKernel,
                           FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ);

}  // namespace oneflow
