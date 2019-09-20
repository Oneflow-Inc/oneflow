#include "oneflow/core/kernel/prelu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  PReluKernelUtil<device_type, T>::Forward(ctx, this->op_conf().prelu_conf(), BnInOp2Blob("in"),
                                           BnInOp2Blob("alpha"), BnInOp2Blob("out"));
}

template<typename T>
struct PReluKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                      const Blob* alpha_blob, Blob* out_blob) {
    const T* in_dptr = in_blob->dptr<T>();
    const T* alpha_dptr = alpha_blob->dptr<T>();
    T* out_dptr = out_blob->mut_dptr<T>();
    const int64_t elem_cnt = in_blob->shape().elem_cnt();
    if (conf.channel_shared()) {
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        out_dptr[i] = (in_dptr[i] >= 0) ? in_dptr[i] : in_dptr[i] * alpha_dptr[0];
      }
    } else {
      if (conf.data_format() == "channels_first") {
        const int64_t channel_num = in_blob->shape().At(1);
        const int64_t area = in_blob->shape().Count(2);
        FOR_RANGE(int64_t, i, 0, elem_cnt) {
          int64_t c = (i / area) % channel_num;
          out_dptr[i] = (in_dptr[i] >= 0) ? in_dptr[i] : in_dptr[i] * alpha_dptr[c];
        }
      } else if (conf.data_format() == "channels_last") {
        const int64_t channel_num = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
        FOR_RANGE(int64_t, i, 0, elem_cnt) {
          int64_t c = i % channel_num;
          out_dptr[i] = (in_dptr[i] >= 0) ? in_dptr[i] : in_dptr[i] * alpha_dptr[c];
        }
      } else {
        UNIMPLEMENTED();
      }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPreluConf, PReluKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
