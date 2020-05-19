#include "oneflow/core/kernel/prelu_alpha_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PReluAlphaGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* alpha_grad_blob = BnInOp2Blob("alpha_grad");
  Memset<device_type>(ctx.device_ctx, alpha_grad_blob->mut_dptr<T>(), 0,
                      alpha_grad_blob->ByteSizeOfBlobBody());
  PReluAlphaGradKernelUtil<device_type, T>::Compute(
      ctx, this->op_conf().prelu_alpha_grad_conf(),
      this->kernel_conf().prelu_alpha_grad_conf().perm(), BnInOp2Blob("x"), BnInOp2Blob("dy"),
      BnInOp2Blob("bw_buf"), BnInOp2Blob("alpha_grad_buf"), alpha_grad_blob);
}

template<typename T>
struct PReluAlphaGradKernelUtil<DeviceType::kCPU, T> {
  static void Compute(const KernelCtx& ctx, const PReluAlphaGradOpConf& conf,
                      const PbRf<int32_t>& permutation, const Blob* x_blob, const Blob* dy_blob,
                      Blob* bw_buf_blob, Blob* alpha_grad_buf_blob, Blob* alpha_grad_blob) {
    const T* x = x_blob->dptr<T>();
    const T* dy = dy_blob->dptr<T>();
    T* alpha_grad_dptr = alpha_grad_blob->mut_dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    if (conf.data_format() == "channels_first") {
      const int64_t channel_num = x_blob->shape().At(1);
      const int64_t alpha_channel_num = conf.channel_shared() ? channel_num : 1;
      const int64_t area = x_blob->shape().Count(2);
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        if (x[i] <= 0) {
          int64_t c = (i / area) % channel_num / alpha_channel_num;
          alpha_grad_dptr[c] += dy[i] * x[i];
        }
      }
    } else if (conf.data_format() == "channels_last") {
      const int64_t channel_num = x_blob->shape().At(x_blob->shape().NumAxes() - 1);
      const int64_t alpha_channel_num = conf.channel_shared() ? channel_num : 1;
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        if (x[i] <= 0) {
          int64_t c = i % channel_num / alpha_channel_num;
          alpha_grad_dptr[c] += dy[i] * x[i];
        }
      }
    } else {
      UNIMPLEMENTED();
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPreluAlphaGradConf, PReluAlphaGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
