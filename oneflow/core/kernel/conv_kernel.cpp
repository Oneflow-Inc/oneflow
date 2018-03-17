#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  WeightForward(ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("weight"),
                BnInOp2Blob("out"), BnInOp2Blob("cudnn_workspace"));
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    BiasForward(ctx.device_ctx, BnInOp2Blob("bias"), BnInOp2Blob("out"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    BiasBackward(ctx.device_ctx, BnInOp2Blob("out_diff"),
                 BnInOp2Blob("bias_diff"));
  }
  WeightBackward(ctx.device_ctx, BnInOp2Blob("out_diff"), BnInOp2Blob("in"),
                 BnInOp2Blob("weight_diff"), BnInOp2Blob("cudnn_workspace"));
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob) {
    DataBackward(ctx.device_ctx, BnInOp2Blob("out_diff"), BnInOp2Blob("weight"),
                 in_diff_blob, BnInOp2Blob("cudnn_workspace"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Initialize(ctx,
                                           bias_multiplier_initializer_conf, 0,
                                           BnInOp2Blob("bias_multiplier"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, this->GetMessageFromCustomizedOpConf("weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("weight"));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx, this->GetMessageFromCustomizedOpConf("bias_initializer"),
        (*random_seed_gen)(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = this->GetInt32FromCustomizedOpConf("filters");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, weight_blob, "weight", dim_num,
      weight_blob->shape().Count(1));
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"), "bias",
        dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& ConvKernelIf<device_type, T>::GetCustomizedOpConf() const {
  CHECK(this->kernel_conf().has_conv_conf());
  switch (KernelDim()) {
    case 1: return this->op_conf().conv_1d_conf();
    case 2: return this->op_conf().conv_2d_conf();
    case 3: return this->op_conf().conv_3d_conf();
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const ConvKernelConf& ConvKernelIf<device_type, T>::GetConvKernelConf() const {
  return this->kernel_conf().conv_conf();
}

template<DeviceType device_type, typename T>
const int32_t ConvKernelIf<device_type, T>::KernelDim() const {
  return GetConvKernelConf().in().dim_size() - 2;
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::WeightForward(
    DeviceCtx* device_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* col_buf_blob = BnInOp2Blob("col_buf");
  FOR_RANGE(int64_t, i, 0, in_blob->shape().At(0)) {
    ConvKernelUtil<T>::Im2Col(device_ctx, in_blob, weight_blob, col_buf_blob);
    // TODO
    KernelUtil<DeviceType::kCPU, T>::Gemm();
  }
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::BiasForward(
    DeviceCtx* device_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bias_blob = BnInOp2Blob("bias");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<DeviceType::kCPU, T>::Axpy(device_ctx, out_blob->shape().elem_cnt(),
                                        static_cast<T>(1), bias_blob->dptr<T>(), 1,
                                        out_blob->mut_dptr<T>(), 1);
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::DataBackward(
    DeviceCtx* device_ctx, const Blob* out_diff, const Blob* weight,
    Blob* in_diff, Blob* cudnn_workspace) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff, const Blob* in,
    Blob* weight_diff, Blob* cudnn_workspace) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::BiasBackward(DeviceCtx* device_ctx,
                                                   const Blob* out_diff,
                                                   Blob* bias_diff) const {
  UNIMPLEMENTED();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv1DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv2DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv3DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);

template<typename T>
class ConvKernelUtil final {
 public:
  static void Im2Col(DeviceCtx* device_ctx, int64_t img_dix, const Blob* in_blob,
                     Blob* col_buf_blob) {
    T* col_buf = col_buf_blob->mut_dptr<T>();
    const T* in = in_blob->dptr<T>();
    int64_t d_size = in_blob->shape().At(2);
    int64_t h_size = in_blob->shape().At(3);
    int64_t w_size = in_blob->shape().At(4);
    int64_t d_dim = in_blob->shape().Count(3);
    int64_t h_dim = in_blob->shape().Count(4);
    for (int64_t c = 0; c != in_blob->shape().At(1); in += channel_size) {
      FOR_RANGE(int64_t, kd, 0, weight_blob->shape().At(1)) {
        FOR_RANGE(int64_t, kh, 0, weight_blob->shape().At(2)) {
          FOR_RANGE(int64_t, kw, 0, weight_blob->shape().At(3)) {
            int64_t id = kd * dilation_rate.Get(0) - padding_before.Get(0);
            for (int64_t od = d_size; od > 0; od--) {
              if (id < 0 || id >= d_size) {
                FOR_RANGE(int64_t, out, 0, d_dim) { *(col_buf++) = 0; }
              } else {
                int64_t ih = kh * dilation_rate.Get(1) - padding_before.Get(1);
                for (int64_t oh = h_size; oh > 0; oh--) {
                  if (ih < 0 || ih >= h_size) {
                    FOR_RANGE(int64_t, out, 0, h_dim) { *(col_buf++) = 0; }
                  } else {
                    int64_t iw = kw * dilation_rate.Get(2) - padding_before.Get(2);
                    for (int64_t ow = w_size; ow > 0; ow--) {
                      if (iw < 0 || iw >= w_size) {
                        *(col_buf++) = 0;
                      } else {
                        *(col_buf++) = in[id * dim_d + ih * dim_h + iw];
                      }
                      iw += strides.At(2);
                    }
                  }
                  ih += strides.At(1);
                }
              }
              id += strides.At(0);
            }
          }
        }
      }
    }
  }
};


}  // namespace oneflow
