#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel<device_type, FloatingPointType>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  switch (op()->op_conf().pooling_conf().pool()) {
    case PoolingOpConf::MAX:
      PoolingMethodForwardFunc_ =
          &PoolingKernelUtil<device_type, FloatingPointType>::RangeMaxQuery;
      PoolingMethodBackwardFunc_ =
          &PoolingKernelUtil<device_type, FloatingPointType>::PoolingMaxBp;
      break;
    case PoolingOpConf::AVE:
      PoolingMethodForwardFunc_ =
          &PoolingKernelUtil<device_type, FloatingPointType>::RangeAveQuery;
      PoolingMethodBackwardFunc_ =
          &PoolingKernelUtil<device_type, FloatingPointType>::PoolingAveBp;
      break;
    case PoolingOpConf::STOCHASTIC:
      PoolingMethodForwardFunc_ =
          &PoolingKernelUtil<device_type, FloatingPointType>::RangeStoQuery;
      PoolingMethodBackwardFunc_ =
          &PoolingKernelUtil<device_type, FloatingPointType>::PoolingStoBp;
      break;
    default: UNEXPECTED_RUN();
  }
}

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* mask_blob = BnInOp2Blob("idx");
  CHECK((in_blob->shape().NumAxes() == 4)
        && (out_blob->shape().NumAxes() == 4));
  const FloatingPointType* in_dptr =
      static_cast<const FloatingPointType*>(in_blob->dptr());
  FloatingPointType* out_dptr =
      static_cast<FloatingPointType*>(out_blob->mut_dptr());
  int64_t* mask_dptr = static_cast<int64_t*>(mask_blob->mut_dptr());

  auto pooling_conf = op()->op_conf().pooling_conf();

  for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
    for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
      for (int64_t out_h = 0; out_h < out_blob->shape().At(2); ++out_h) {
        for (int64_t out_w = 0; out_w < out_blob->shape().At(3); ++out_w) {
          int64_t hstart = out_h * pooling_conf.stride(0) - pooling_conf.pad(0);
          int64_t wstart = out_w * pooling_conf.stride(1) - pooling_conf.pad(1);
          int64_t hend = std::min(hstart + pooling_conf.kernel_size(0),
                                  in_blob->shape().At(2) + pooling_conf.pad(0));
          int64_t wend = std::min(wstart + pooling_conf.kernel_size(1),
                                  in_blob->shape().At(3) + pooling_conf.pad(1));
          int64_t pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          hend = std::min(hend, in_blob->shape().At(2));
          wend = std::min(wend, in_blob->shape().At(3));
          const int64_t out_index = out_h * out_blob->shape().At(3) + out_w;
          PoolingMethodForwardFunc_(ctx, in_dptr, in_blob->shape().At(2),
                                    in_blob->shape().At(3), pool_size, hstart,
                                    wstart, hend, wend, out_index, out_dptr,
                                    mask_dptr);
        }
      }
      in_dptr += in_blob->shape().Count(2);
      out_dptr += out_blob->shape().Count(2);
      mask_dptr += out_blob->shape().Count(2);
    }
  }
}

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in_diff") == nullptr) { return; }

  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* mask_blob = BnInOp2Blob("idx");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  CHECK((in_diff_blob->shape().NumAxes() == 4)
        && (out_diff_blob->shape().NumAxes() == 4));

  const FloatingPointType* out_diff_dptr =
      static_cast<const FloatingPointType*>(out_diff_blob->dptr());
  const int64_t* mask_dptr = static_cast<const int64_t*>(mask_blob->dptr());
  FloatingPointType* in_diff_dptr =
      static_cast<FloatingPointType*>(in_diff_blob->mut_dptr());

  auto pooling_conf = op()->op_conf().pooling_conf();

  for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
    for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
      for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2); ++out_h) {
        for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3); ++out_w) {
          int64_t hstart = out_h * pooling_conf.stride(0) - pooling_conf.pad(0);
          int64_t wstart = out_w * pooling_conf.stride(1) - pooling_conf.pad(1);
          int64_t hend =
              std::min(hstart + pooling_conf.kernel_size(0),
                       in_diff_blob->shape().At(2) + pooling_conf.pad(0));
          int64_t wend =
              std::min(wstart + pooling_conf.kernel_size(1),
                       in_diff_blob->shape().At(3) + pooling_conf.pad(1));
          int64_t pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          hend = std::min(hend, in_diff_blob->shape().At(2));
          wend = std::min(wend, in_diff_blob->shape().At(3));
          const int64_t out_diff_index =
              out_h * out_diff_blob->shape().At(3) + out_w;
          PoolingMethodBackwardFunc_(
              ctx, out_diff_dptr, mask_dptr, pool_size, out_diff_index,
              in_diff_blob->shape().At(2), in_diff_blob->shape().At(3), hstart,
              wstart, hend, wend, in_diff_dptr);
        }
      }
      out_diff_dptr +=
          out_diff_blob->shape().Count(2);
      mask_dptr += out_diff_blob->shape().Count(2);
      in_diff_dptr += in_diff_blob->shape().Count(2);
    }
  }
}

template<typename FloatingPointType>
class PoolingKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void RangeMaxQuery(const KernelCtx& ctx, const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      out_dptr[out_index] = in_dptr[hstart * in_width + wstart];
      mask_dptr[out_index] = hstart * in_width + wstart;
      for (int64_t h = hstart; h < hend; ++h) {
        for (int64_t w = wstart; w < wend; ++w) {
          const int64_t index = h * in_width + w;
          if (in_dptr[index] > out_dptr[out_index]) {
            out_dptr[out_index] = in_dptr[index];
            mask_dptr[out_index] = index;
          }
        }
      }
    });
  }

  static void RangeAveQuery(const KernelCtx& ctx, const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      out_dptr[out_index] = 0;
      for (int64_t h = hstart; h < hend; ++h) {
        for (int64_t w = wstart; w < wend; ++w) {
          const int64_t index = h * in_width + w;
          out_dptr[out_index] += in_dptr[index];
        }
      }
      out_dptr[out_index] /= pool_size;
    });
  }

  static void RangeStoQuery(const KernelCtx& ctx, const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      std::mt19937 generator(KernelUtil<device_type, FloatingPointType>::NewRandomSeed());
      const int64_t index = (hstart + (generator() % (hend - hstart))) * in_width + (wstart + (generator() % (wend - wstart)));
      out_dptr[out_index] = in_dptr[index];
      mask_dptr[out_index] = index;
    });
  }

  static void PoolingMaxBp(const KernelCtx& ctx, const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      const int64_t in_diff_index = mask_dptr[out_diff_index];
      in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
    });
  }

  static void PoolingAveBp(const KernelCtx& ctx, const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          in_diff_dptr[h * in_width + w] +=
              out_diff_dptr[out_diff_index] / pool_size;
        }
      }
    });
  }

  static void PoolingStoBp(const KernelCtx& ctx, const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() mutable {
      const int64_t in_diff_index = mask_dptr[out_diff_index];
      in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(PoolingKernelUtil);
INSTANTIATE_KERNEL_CLASS(PoolingKernel);
REGISTER_KERNEL(OperatorConf::kPoolingConf, PoolingKernel);

}  // namespace oneflow
