#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template<typename FloatingPointType>
void Vec2Matrix(const KernelCtx& ctx, const FloatingPointType* data_col,
                const int channels, const int height, const int width,
                const int kernel_h, const int kernel_w, const int pad_h,
                const int pad_w, const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                FloatingPointType* mut_dptr) {
  ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
    memset(mut_dptr, 0, height * width * channels);
    const int output_h =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; mut_dptr += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              data_col += output_w;
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  mut_dptr[input_row * width + input_col] += *data_col;
                }
                data_col++;
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
  });
}

template<typename FloatingPointType>
void Matrix2Vec(const KernelCtx& ctx, const FloatingPointType* dptr,
                const int channels, const int height, const int width,
                const int kernel_h, const int kernel_w, const int pad_h,
                const int pad_w, const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                FloatingPointType* data_col) {
  ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
    const int output_h =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; dptr += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = dptr[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
  });
}

}  // namespace

template<typename FloatingPointType>
void ConvolutionKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  Blob* col_buf = BnInOp2Blob("col_buf");
  Blob* weight = BnInOp2Blob("weight");
  const Shape& in_shape = in->shape();
  CHECK_EQ(in_shape.NumAxes(), 4);
  const int64_t in_im_sz = in_shape.Count(1) * sizeof(FloatingPointType);
  const int64_t out_im_sz = out->shape().Count(1) * sizeof(FloatingPointType);
  const int64_t col_im_sz =
      col_buf->shape().Count(1) * sizeof(FloatingPointType);
  auto conv_conf = op()->op_conf().convolution_conf();
  for (size_t i = 0; i < in_shape.At(0); ++i) {
    Matrix2Vec<FloatingPointType>(
        ctx, static_cast<FloatingPointType*>(in->mut_dptr()) + i * in_im_sz,
        in_shape.At(1), in_shape.At(2), in_shape.At(3),
        conv_conf.kernel_size(2), conv_conf.kernel_size(3), conv_conf.pad(2),
        conv_conf.pad(3), conv_conf.stride(2), conv_conf.stride(3),
        conv_conf.dilation(2), conv_conf.dilation(3),
        static_cast<FloatingPointType*>(col_buf->mut_dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType));

    KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasGemm(
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
      KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasGemm(
          ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
          bias->shape().At(0), bias_multiplier->shape().At(0), 1,
          static_cast<FloatingPointType>(1.0),
          static_cast<const FloatingPointType*>(bias->dptr()), 1,
          static_cast<const FloatingPointType*>(bias_multiplier->dptr()), 1,
          static_cast<FloatingPointType>(1.0),
          static_cast<FloatingPointType*>(out->mut_dptr()) + i * out_im_sz,
          bias_multiplier->shape().At(0));
    }
  }
}

template<typename FloatingPointType>
void ConvolutionKernel<DeviceType::kCPU, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob("out_diff");
  const Blob* weight = BnInOp2Blob("weight");
  Blob* col_buf = BnInOp2Blob("col_buf");

  const int64_t out_im_sz =
      out_diff->shape().Count(1) * sizeof(FloatingPointType);
  int64_t batch_sz = out_diff->shape().At(0);

  // compute weight_diff
  Blob* weight_diff = BnInOp2Blob("weight_diff");
  for (size_t i = 0; i < batch_sz; ++i) {
    KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasGemm(
        ctx, CBLAS_ORDER::CblasRowMajor, CblasNoTrans, CblasNoTrans,
        weight_diff->shape().At(1), weight_diff->shape().At(2),
        out_diff->shape().At(2), static_cast<FloatingPointType>(1.0),
        static_cast<const FloatingPointType*>(out_diff->dptr()) + i * out_im_sz,
        out_diff->shape().Count(2),
        static_cast<const FloatingPointType*>(col_buf->dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType),
        col_buf->shape().At(2), static_cast<FloatingPointType>(1.0),
        static_cast<FloatingPointType*>(weight_diff->mut_dptr()),
        weight_diff->shape().At(1));
  }

  // compute bias_diff
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    const Blob* bias_multiplier = BnInOp2Blob("bias_multiplier");
    Blob* bias_diff = BnInOp2Blob("bias_diff");
    for (size_t i = 0; i < batch_sz; ++i) {
      KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasGemm(
          ctx, CBLAS_ORDER::CblasRowMajor, CblasTrans, CblasNoTrans,
          bias_diff->shape().At(0), 1, bias_multiplier->shape().At(0),
          static_cast<FloatingPointType>(1.0),
          static_cast<const FloatingPointType*>(out_diff->dptr())
              + i * out_diff->shape().Count(2) * sizeof(FloatingPointType),
          out_diff->shape().Count(2),
          static_cast<const FloatingPointType*>(bias_multiplier->dptr()), 1,
          static_cast<FloatingPointType>(1.0),
          static_cast<FloatingPointType*>(bias_diff->mut_dptr()), 1);
    }
  }

  // compute in_diff
  for (size_t i = 0; i < batch_sz; ++i) {
    KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasGemm(
        ctx, CBLAS_ORDER::CblasRowMajor, CblasTrans, CblasNoTrans,
        col_buf->shape().At(1), col_buf->shape().At(2), weight->shape().At(1),
        static_cast<FloatingPointType>(1.0),
        static_cast<const FloatingPointType*>(out_diff->dptr()) + i * out_im_sz,
        out_diff->shape().At(1),
        static_cast<const FloatingPointType*>(weight->dptr()),
        weight->shape().At(1), static_cast<FloatingPointType>(0.0),
        static_cast<FloatingPointType*>(col_buf->mut_dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType),
        col_buf->shape().At(1));
  }

  Blob* in_diff = BnInOp2Blob("in_diff");
  const Shape& in_diff_shape = in_diff->shape();
  auto conv_conf = op()->op_conf().convolution_conf();
  for (size_t i = 0; i < batch_sz; ++i) {
    Vec2Matrix<FloatingPointType>(
        ctx,
        static_cast<FloatingPointType*>(col_buf->mut_dptr())
            + i * col_buf->shape().Count(1) * sizeof(FloatingPointType),
        in_diff_shape.At(1), in_diff_shape.At(2), in_diff_shape.At(3),
        conv_conf.kernel_size(2), conv_conf.kernel_size(3), conv_conf.pad(2),
        conv_conf.pad(3), conv_conf.stride(2), conv_conf.stride(3),
        conv_conf.dilation(2), conv_conf.dilation(3),
        static_cast<FloatingPointType*>(in_diff->mut_dptr())
            + i * in_diff_shape.Count(1) * sizeof(FloatingPointType));
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(ConvolutionKernel);
REGISTER_CPU_KERNEL(OperatorConf::kConvolutionConf, ConvolutionKernel);

}  // namespace oneflow
