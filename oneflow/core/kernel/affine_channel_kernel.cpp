#include "oneflow/core/kernel/affine_channel_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& AffineChannelKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().affine_channel_conf();
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().affine_channel_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* scale_blob = BnInOp2Blob("scale");
  const Blob* bias_blob = BnInOp2Blob("bias");
  Blob* out_blob = BnInOp2Blob("out");
  const int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
  const int32_t channel_dim = in_blob->shape().At(axis);
  const int64_t channel_stride = in_blob->shape().Count(axis + 1);
  const T* bias_ptr = conf.use_bias() ? bias_blob->dptr<T>() : nullptr;
  AffineChannelKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, in_blob->shape().elem_cnt(), channel_dim, channel_stride, in_blob->dptr<T>(),
      scale_blob->dptr<T>(), bias_ptr, out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().affine_channel_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* scale_blob = BnInOp2Blob("scale");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* scale_diff_blob = BnInOp2Blob("scale_diff");
  Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
  const int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
  const int32_t channel_dim = out_diff_blob->shape().At(axis);
  const int64_t channel_stride = out_diff_blob->shape().Count(axis + 1);
  T* bias_diff_ptr = conf.use_bias() ? bias_diff_blob->mut_dptr<T>() : nullptr;
  AffineChannelKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, out_diff_blob->shape().elem_cnt(), channel_dim, channel_stride,
      in_blob->dptr<T>(), out_diff_blob->dptr<T>(), scale_blob->dptr<T>(),
      in_diff_blob->mut_dptr<T>(), scale_diff_blob->mut_dptr<T>(), bias_diff_ptr);
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, GetMsgPtrFromPbMessage(this->op_conf().affine_channel_conf(), "scale_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("scale"));
  if (this->op_conf().affine_channel_conf().use_bias()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx, GetMsgPtrFromPbMessage(this->op_conf().affine_channel_conf(), "bias_initializer"),
        (*random_seed_gen)(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* scale_blob = BnInOp2Blob("scale");
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, scale_blob,
                                                "scale", scale_blob->shape().At(0),
                                                scale_blob->shape().Count(1));
  if (this->op_conf().affine_channel_conf().use_bias()) {
    Blob* bias_blob = BnInOp2Blob("bias");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, bias_blob,
                                                  "bias", bias_blob->shape().At(0),
                                                  bias_blob->shape().Count(1));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAffineChannelConf, AffineChannelKernel,
                           FLOATING_DATA_TYPE_SEQ);

template<typename T>
class AffineChannelKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const int32_t channel_dim,
                      const int64_t channel_stride, const T* in, const T* scale, const T* bias,
                      T* out) {
    for (int64_t i = 0; i < elem_cnt; ++i) {
      const int32_t channel_i = (i / channel_stride) % channel_dim;
      if (bias != nullptr) {
        out[i] = in[i] * scale[channel_i] + bias[channel_i];
      } else {
        out[i] = in[i] * scale[channel_i];
      }
    }
  }

  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const int32_t channel_dim,
                       const int64_t channel_stride, const T* in, const T* out_diff, const T* scale,
                       T* in_diff, T* scale_diff, T* bias_diff) {
    // in_diff
    for (int64_t i = 0; i < elem_cnt; ++i) {
      const int32_t channel_i = (i / channel_stride) % channel_dim;
      in_diff[i] = out_diff[i] * scale[channel_i];
    }

    // scale_diff & bias_diff
    for (int32_t i = 0; i < channel_dim; ++i) {
      for (int64_t j = 0; j < (elem_cnt / channel_dim); ++j) {
        int64_t index =
            ((j / channel_stride) * channel_dim + i) * channel_stride + j % channel_stride;
        scale_diff[i] += out_diff[index] * in[index];
        if (bias_diff != nullptr) { bias_diff[i] += out_diff[index]; }
      }
    }
  }
};

#define INSTANTIATE_CPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AffineChannelKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
