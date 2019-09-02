#include "oneflow/core/kernel/prelu_data_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PReluDataGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* dx_blob = BnInOp2Blob("dx");
  if (dx_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                      dx_blob->ByteSizeOfDataContentField());
  PReluDataGradKernelUtil<device_type, T>::Compute(ctx, this->op_conf().prelu_data_grad_conf(),
                                                   BnInOp2Blob("x"), BnInOp2Blob("alpha"),
                                                   BnInOp2Blob("dy"), dx_blob);
}

template<typename T>
struct PReluDataGradKernelUtil<DeviceType::kCPU, T> {
  static void Compute(const KernelCtx& ctx, const PReluDataGradOpConf& conf, const Blob* x_blob,
                      const Blob* alpha_blob, const Blob* dy_blob, Blob* dx_blob) {
    const T* x = x_blob->dptr<T>();
    const T* alpha_dptr = alpha_blob->dptr<T>();
    const T* dy = dy_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    if (conf.data_format() == "channels_first") {
      const int64_t channel_num = x_blob->shape().At(1);
      const int64_t alpha_channel_num = conf.channel_shared() ? channel_num : 1;
      const int64_t area = x_blob->shape().Count(2);
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        if (x[i] > 0) {
          dx[i] = dy[i];
        } else {
          int64_t c = (i / area) % channel_num / alpha_channel_num;
          dx[i] = alpha_dptr[c] * dy[i];
        }
      }
    } else if (conf.data_format() == "channels_last") {
      const int64_t channel_num = x_blob->shape().At(x_blob->shape().NumAxes() - 1);
      const int64_t alpha_channel_num = conf.channel_shared() ? channel_num : 1;
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        if (x[i] > 0) {
          dx[i] = dy[i];
        } else {
          int64_t c = i % channel_num / alpha_channel_num;
          dx[i] = alpha_dptr[c] * dy[i];
        }
      }
    } else {
      UNIMPLEMENTED();
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPreluDataGradConf, PReluDataGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
