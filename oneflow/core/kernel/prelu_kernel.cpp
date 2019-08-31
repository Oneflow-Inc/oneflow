#include "oneflow/core/kernel/prelu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  PReluKernelUtil<device_type, T>::Forward(ctx, this->op_conf().prelu_conf(), BnInOp2Blob("in"),
                                           BnInOp2Blob("alpha"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* alpha_diff_blob = BnInOp2Blob("alpha_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  Memset<device_type>(ctx.device_ctx, alpha_diff_blob->mut_dptr<T>(), 0,
                      alpha_diff_blob->ByteSizeOfDataContentField());
  PReluKernelUtil<device_type, T>::Backward(
      ctx, this->op_conf().prelu_conf(), this->kernel_conf().prelu_conf().perm(), BnInOp2Blob("in"),
      BnInOp2Blob("alpha"), BnInOp2Blob("out_diff"), BnInOp2Blob("bw_buf"), in_diff_blob,
      alpha_diff_blob);
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
  static void Backward(const KernelCtx& ctx, const PReluOpConf& conf,
                       const PbRf<int32_t>& permutation, const Blob* in_blob,
                       const Blob* alpha_blob, const Blob* out_diff_blob, Blob* bw_buf_blob,
                       Blob* in_diff_blob, Blob* alpha_diff_blob) {
    const T* in_dptr = in_blob->dptr<T>();
    const T* alpha_dptr = alpha_blob->dptr<T>();
    const T* out_diff_dptr = out_diff_blob->dptr<T>();
    T* in_diff_dptr = in_diff_blob->mut_dptr<T>();
    T* alpha_diff_dptr = alpha_diff_blob->mut_dptr<T>();
    const int64_t elem_cnt = in_blob->shape().elem_cnt();
    if (conf.data_format() == "channels_first") {
      const int64_t channel_num = in_blob->shape().At(1);
      const int64_t alpha_channel_num = conf.channel_shared() ? channel_num : 1;
      const int64_t area = in_blob->shape().Count(2);
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        if (in_dptr[i] <= 0) {
          int64_t c = (i / area) % channel_num / alpha_channel_num;
          alpha_diff_dptr[c] += out_diff_dptr[i] * in_dptr[i];
        }
        if (in_dptr[i] > 0) {
          in_diff_dptr[i] = out_diff_dptr[i];
        } else {
          int64_t c = (i / area) % channel_num / alpha_channel_num;
          in_diff_dptr[i] = alpha_dptr[c] * out_diff_dptr[i];
        }
      }
    } else if (conf.data_format() == "channels_last") {
      const int64_t channel_num = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
      const int64_t alpha_channel_num = conf.channel_shared() ? channel_num : 1;
      FOR_RANGE(int64_t, i, 0, elem_cnt) {
        if (in_dptr[i] <= 0) {
          int64_t c = i % channel_num / alpha_channel_num;
          alpha_diff_dptr[c] += out_diff_dptr[i] * in_dptr[i];
        }
        if (in_dptr[i] > 0) {
          in_diff_dptr[i] = out_diff_dptr[i];
        } else {
          int64_t c = i % channel_num / alpha_channel_num;
          in_diff_dptr[i] = alpha_dptr[c] * out_diff_dptr[i];
        }
      }
    } else {
      UNIMPLEMENTED();
    }
  }
};

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kPreluConf, PReluKernel);

}  // namespace oneflow
