#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void ConvolutionKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in = BnInOp2Blob("in");
  const Shape& in_shape = in->shape();
  CHECK_EQ(in_shape.NumAxes(), 4);
  Blob* out = BnInOp2Blob("out");
  Blob* col_buf = BnInOp2Blob("col_buf");
  Blob* weight = BnInOp2Blob("weight");
  const int64_t in_im_sz = in_shape.Count(1) * sizeof(FloatingPointType);
  const int64_t out_im_sz = out->shape().Count(1) * sizeof(FloatingPointType);
  const int64_t col_im_sz =
      col_buf->shape().Count(1) * sizeof(FloatingPointType);
  auto conv_conf = op()->op_conf().convolution_conf();
  for (size_t i = 0; i < in_shape.At(0); ++i) {
    KernelUtil<device_type, FloatingPointType>::im2col(
        ctx, static_cast<FloatingPointType*>(in->mut_dptr()) + i * in_im_sz,
        in_shape.At(1), in_shape.At(2), in_shape.At(3),
        conv_conf.kernel_size(0), conv_conf.kernel_size(1), conv_conf.pad(0),
        conv_conf.pad(1), conv_conf.stride(0), conv_conf.stride(1),
        conv_conf.dilation(0), conv_conf.dilation(1),
        static_cast<FloatingPointType*>(col_buf->mut_dptr()) + i * col_im_sz);

    KernelUtil<device_type, FloatingPointType>::BlasGemm(
        ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasTrans,
        out->shape().At(1), out->shape().Count(2), weight->shape().At(1),
        static_cast<FloatingPointType>(1.0),
        static_cast<const FloatingPointType*>(weight->dptr()),
        weight->shape().At(1),
        static_cast<const FloatingPointType*>(col_buf->dptr()) + i * col_im_sz,
        weight->shape().At(1), static_cast<FloatingPointType>(1.0),
        static_cast<FloatingPointType*>(out->mut_dptr()) + i * out_im_sz,
        col_buf->shape().At(1));

    if (op()->GetBoolFromSpecialConf("has_bias_term")) {
      const Blob* bias = BnInOp2Blob("bias");
      const Blob* bias_multiplier = BnInOp2Blob("bias_multiplier");

      // out_data = bias * bias_multiplier + out_data
      KernelUtil<device_type, FloatingPointType>::BlasGemm(
          ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
          bias->shape().At(0), bias_multiplier->shape().At(0), 1,
          static_cast<FloatingPointType>(1.0),
          static_cast<const FloatingPointType*>(bias->dptr()), 1,
          static_cast<const FloatingPointType*>(bias_multiplier->dptr()),
          bias_multiplier->shape().At(0), static_cast<FloatingPointType>(1.0),
          static_cast<FloatingPointType*>(out->mut_dptr()) + i * out_im_sz,
          bias_multiplier->shape().At(0));
    }
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConvolutionKernel<device_type, FloatingPointType>::ComputeWeightDiff(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_diff = BnInOp2Blob("weight_diff");
  Blob* col_buf = BnInOp2Blob("col_buf");
  const Blob* out_diff = BnInOp2Blob("out_diff");
  const int64_t out_im_sz =
      out_diff->shape().Count(1) * sizeof(FloatingPointType);
  int64_t batch_sz = out_diff->shape().At(0);

  KernelUtil<device_type, FloatingPointType>::Memset(
      ctx, weight_diff->mut_dptr(), 0,
      sizeof(FloatingPointType) * weight_diff->shape().elem_cnt());
  for (size_t i = 0; i < batch_sz; ++i) {
    KernelUtil<device_type, FloatingPointType>::BlasGemm(
        ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
        weight_diff->shape().At(0), weight_diff->shape().At(1),
        out_diff->shape().Count(2), static_cast<FloatingPointType>(1.0),
        static_cast<const FloatingPointType*>(out_diff->dptr()) + i * out_im_sz,
        out_diff->shape().Count(2),
        static_cast<const FloatingPointType*>(col_buf->dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType),
        col_buf->shape().At(2), static_cast<FloatingPointType>(1.0),
        static_cast<FloatingPointType*>(weight_diff->mut_dptr()),
        weight_diff->shape().At(1));
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConvolutionKernel<device_type, FloatingPointType>::ComputeBiasDiff(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob("out_diff");
  const int64_t out_im_sz =
      out_diff->shape().Count(1) * sizeof(FloatingPointType);
  int64_t batch_sz = out_diff->shape().At(0);

  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    const Blob* bias_mul = BnInOp2Blob("bias_multiplier");
    Blob* bias_diff = BnInOp2Blob("bias_diff");
    KernelUtil<device_type, FloatingPointType>::Memset(
        ctx, bias_diff->mut_dptr(), 0,
        sizeof(FloatingPointType) * bias_diff->shape().elem_cnt());
    for (size_t i = 0; i < batch_sz; ++i) {
      KernelUtil<device_type, FloatingPointType>::BlasGemm(
          ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
          bias_diff->shape().At(0), 1, bias_mul->shape().At(0),
          static_cast<FloatingPointType>(1.0),
          static_cast<const FloatingPointType*>(out_diff->dptr())
              + i * out_im_sz,
          out_diff->shape().Count(2),
          static_cast<const FloatingPointType*>(bias_mul->dptr()), 1,
          static_cast<FloatingPointType>(1.0),
          static_cast<FloatingPointType*>(bias_diff->mut_dptr()), 1);
    }
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConvolutionKernel<device_type, FloatingPointType>::ComputeInputDiff(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob("out_diff");
  const Blob* weight = BnInOp2Blob("weight");
  Blob* col_buf = BnInOp2Blob("col_buf");

  const int64_t out_im_sz =
      out_diff->shape().Count(1) * sizeof(FloatingPointType);
  int64_t batch_sz = out_diff->shape().At(0);
  for (size_t i = 0; i < batch_sz; ++i) {
    KernelUtil<device_type, FloatingPointType>::BlasGemm(
        ctx, CBLAS_ORDER::CblasRowMajor, CblasTrans, CblasNoTrans,
        col_buf->shape().At(1), col_buf->shape().At(2), weight->shape().At(0),
        static_cast<FloatingPointType>(1.0),
        static_cast<const FloatingPointType*>(out_diff->dptr()) + i * out_im_sz,
        out_diff->shape().Count(2),
        static_cast<const FloatingPointType*>(weight->dptr()),
        weight->shape().At(1), static_cast<FloatingPointType>(0.0),
        static_cast<FloatingPointType*>(col_buf->mut_dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType),
        col_buf->shape().At(2));
  }

  Blob* in_diff = BnInOp2Blob("in_diff");
  const Shape& in_diff_shape = in_diff->shape();
  auto conv_conf = op()->op_conf().convolution_conf();
  for (size_t i = 0; i < batch_sz; ++i) {
    KernelUtil<device_type, FloatingPointType>::col2im(
        ctx,
        static_cast<const FloatingPointType*>(col_buf->dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType),
        in_diff_shape.At(1), in_diff_shape.At(2), in_diff_shape.At(3),
        conv_conf.kernel_size(0), conv_conf.kernel_size(1), conv_conf.pad(0),
        conv_conf.pad(1), conv_conf.stride(0), conv_conf.stride(1),
        conv_conf.dilation(0), conv_conf.dilation(1),
        static_cast<FloatingPointType*>(in_diff->mut_dptr())
            + i * in_diff_shape.Count(1) * sizeof(FloatingPointType));
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConvolutionKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ComputeWeightDiff(ctx, BnInOp2Blob);
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    ComputeBiasDiff(ctx, BnInOp2Blob);
  }
  ComputeInputDiff(ctx, BnInOp2Blob);
}

INSTANTIATE_KERNEL_CLASS(ConvolutionKernel);
REGISTER_CPU_KERNEL(OperatorConf::kConvolutionConf, ConvolutionKernel);

}  // namespace oneflow
