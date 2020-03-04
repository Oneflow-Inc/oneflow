#include "oneflow/core/kernel/maximum_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void MaximumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob_0 = BnInOp2Blob(this->op_attribute().input_bns(0));
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob_0);
  Blob* mask_blob = BnInOp2Blob("mask");
  Memset<device_type>(ctx.device_ctx, mask_blob->mut_dptr(), 0, mask_blob->ByteSizeOfBlobBody());
  const int64_t elem_cnt = out_blob->shape().elem_cnt();
  FOR_RANGE(size_t, i, 1, this->op_attribute().input_bns().size()) {
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(i));
    MaximumKernelUtil<device_type, T>::CWiseMaxWithMask(ctx.device_ctx, elem_cnt,
                                                        out_blob->mut_dptr<T>(), in_blob->dptr<T>(),
                                                        i, mask_blob->mut_dptr<int32_t>());
  }
}

template<typename T>
struct MaximumKernelUtil<DeviceType::kCPU, T> {
  static void CWiseMaxWithMask(DeviceCtx* ctx, const int64_t n, T* x, const T* y, const int y_idx,
                               int32_t* mask) {
    for (int64_t i = 0; i < n; ++i) {
      if (y[i] > x[i]) {
        x[i] = y[i];
        mask[i] = y_idx;
      }
    }
  }

  static void CWiseSetWithMask(DeviceCtx* ctx, const int64_t n, T* x, const T* y, const int x_idx,
                               const int32_t* mask) {
    for (int i = 0; i < n; ++i) {
      if (x_idx == mask[i]) { x[i] = y[i]; }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaximumConf, MaximumKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
