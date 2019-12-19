#include "oneflow/core/kernel/l2_normalize_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("in"),
      BnInOp2Blob("square_x_sum"), BnInOp2Blob("out"));
}

template<typename T>
struct L2NormalizeKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* square_x_sum_blob, Blob* out_blob) {
    const int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
    const int32_t c = in_blob->shape().At(axis);
    const int32_t n = in_blob->shape().elem_cnt() / c;
    const int32_t d = in_blob->shape().Count(axis + 1);
    const T* in = in_blob->dptr<T>();
    T* square_x_sum = square_x_sum_blob->mut_dptr<T>();
    T* out = out_blob->mut_dptr<T>();

    for (int32_t i = 0; i < n; i++) {
      const int32_t offset = (i / d) * d * c + (i % d);
      for (int32_t j = 0; j < c; j++) {
        const T x = in[offset + j * d];
        square_x_sum[i] += x * x;
      }
      const T norm = std::sqrt(std::max(square_x_sum[i], static_cast<T>(conf.epsilon())));
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        out[index] = in[index] / norm;
      }
    }
  }

  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* out_blob,
                       const Blob* out_diff_blob, const Blob* square_x_sum_blob,
                       Blob* in_diff_blob) {
    const int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + out_blob->shape().NumAxes();
    const int32_t c = out_blob->shape().At(axis);
    const int32_t n = out_blob->shape().elem_cnt() / c;
    const int32_t d = out_blob->shape().Count(axis + 1);
    const T* out_diff = out_diff_blob->dptr<T>();
    const T* out = out_blob->dptr<T>();
    const T* square_x_sum = square_x_sum_blob->dptr<T>();
    T* in_diff = in_diff_blob->mut_dptr<T>();

    for (int32_t i = 0; i < n; i++) {
      const T norm = std::sqrt(std::max(square_x_sum[i], static_cast<T>(conf.epsilon())));
      const int32_t offset = (i / d) * d * c + (i % d);
      if (square_x_sum[i] >= conf.epsilon()) {
        T y_dy_inner_prod = GetZeroVal<T>();
        for (int32_t j = 0; j < c; j++) {
          const int32_t index = offset + j * d;
          y_dy_inner_prod += out_diff[index] * out[index];
        }
        for (int32_t j = 0; j < c; j++) {
          const int32_t index = offset + j * d;
          in_diff[index] = (1 / norm) * (out_diff[index] - y_dy_inner_prod * out[index]);
        }
      } else {
        for (int32_t j = 0; j < c; j++) {
          const int32_t index = offset + j * d;
          in_diff[index] = (1 / norm) * out_diff[index];
        }
      }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL2NormalizeConf, L2NormalizeKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
