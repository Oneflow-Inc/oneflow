#include "oneflow/core/kernel/constant_like_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConstantLikeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  ConstantLikeUtil<device_type, T>::Forward(
      ctx.device_ctx, out_blob->shape().elem_cnt(),
      static_cast<T>(this->op_conf().constant_like_conf().scalar()), out_blob->mut_dptr<T>());
}

template<typename T>
struct ConstantLikeUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, T* out_ptr) {
    FOR_RANGE(size_t, i, 0, elem_cnt) { out_ptr[i] = scalar; }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConstantLikeConf, ConstantLikeKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
