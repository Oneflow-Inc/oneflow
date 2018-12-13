#include "oneflow/core/kernel/l2_normalize_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("in"), BnInOp2Blob("norm"),
      BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("out"),
      BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob("norm"), BnInOp2Blob(GenDiffBn("in")));
}

template<typename T>
struct L2NormalizeKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* norm_blob, Blob* out_blob) {
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
    int32_t c = in_blob->shape().At(axis);
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().Count(axis + 1);
    const T* in = in_blob->dptr<T>();
    T* norm = norm_blob->mut_dptr<T>();
    T* out = out_blob->mut_dptr<T>();

    for (int32_t i = 0; i < n; i++) {
      T square_x_sum = ZeroVal<T>::value;
      int32_t offset = (i / d) * d * c + (i % d);
      for (int32_t j = 0; j < c; j++) {
        const T x = in[offset + j * d];
        square_x_sum += x * x;
      }
      norm[i] = std::max(static_cast<T>(conf.epsilon()), std::sqrt(square_x_sum));
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        out[index] = in[index] / norm[i];
      }
    }
  }

  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* out_blob,
                       const Blob* out_diff_blob, const Blob* norm_blob, Blob* in_diff_blob) {
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + out_blob->shape().NumAxes();
    int32_t c = out_blob->shape().At(axis);
    int32_t n = out_blob->shape().elem_cnt() / c;
    int32_t d = out_blob->shape().Count(axis + 1);
    const T* out_diff = out_diff_blob->dptr<T>();
    const T* out = out_blob->dptr<T>();
    const T* norm = norm_blob->dptr<T>();
    T* in_diff = in_diff_blob->mut_dptr<T>();

    for (int32_t i = 0; i < n; i++) {
      T y_dy_inner_prod = ZeroVal<T>::value;
      int32_t offset = (i / d) * d * c + (i % d);
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        y_dy_inner_prod += out_diff[index] * out[index];
      }
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        in_diff[index] = (1 / norm[i]) * (out_diff[index] - y_dy_inner_prod * out[index]);
      }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL2NormalizeConf, L2NormalizeKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
