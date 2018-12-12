#include "oneflow/core/kernel/l2_normalize_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("in"),
      BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob(GenDiffBn("in")));
}

template<typename T>
struct L2NormalizeKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* out_blob) {
    int32_t c = in_blob->shape().At(conf.axis());
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().elem_cnt() / in_blob->shape().Count(0, conf.axis() + 1);
    const T* in = in_blob->dptr<T>();
    float epsilon = conf.epsilon();
    T* out = out_blob->mut_dptr<T>();

    for (int32_t i = 0; i < n; i++) {
      T square_x_sum = ZeroVal<T>::value;
      int32_t beg = (i / d) * d * c + (i % d);
      for (int32_t j = 0; j < c; j++) {
        const T x = in[beg + j * d];
        square_x_sum += x * x;
      }
      const T norm = std::max(static_cast<T>(epsilon), std::sqrt(square_x_sum));
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = beg + j * d;
        out[index] = in[index] / norm;
      }
    }
  }

  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    int32_t c = in_blob->shape().At(conf.axis());
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().elem_cnt() / in_blob->shape().Count(0, conf.axis() + 1);
    const T* out_diff = out_diff_blob->dptr<T>();
    const T* in = in_blob->dptr<T>();
    float epsilon = conf.epsilon();
    T* in_diff = in_diff_blob->mut_dptr<T>();

    for (int32_t i = 0; i < n; i++) {
      T square_x_sum = ZeroVal<T>::value;
      T dy_x_inner_prod = ZeroVal<T>::value;
      int32_t beg = (i / d) * d * c + (i % d);
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = beg + j * d;
        const T x = in[index];
        square_x_sum += x * x;
        dy_x_inner_prod += out_diff[index] * in[index];
      }
      const T norm = std::max(static_cast<T>(epsilon), std::sqrt(square_x_sum));
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = beg + j * d;
        in_diff[index] =
            (out_diff[index] / norm) - (dy_x_inner_prod * in[index] / std::pow(norm, 3));
      }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL2NormalizeConf, L2NormalizeKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow