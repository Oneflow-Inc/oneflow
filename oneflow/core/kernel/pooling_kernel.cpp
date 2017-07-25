#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto pooling_conf = op()->op_conf().pooling_conf();

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* mask_blob = BnInOp2Blob("idx");

  CHECK((in_blob->shape().NumAxes() == 4)
        && (out_blob->shape().NumAxes() == 4));

  PoolingKernelUtil<device_type, FloatingPointType>::PoolingForward(
      ctx, in_blob, out_blob, mask_blob, pooling_conf);
}

template<DeviceType device_type, typename FloatingPointType>
void PoolingKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in_diff") == nullptr) { return; }

  auto pooling_conf = op()->op_conf().pooling_conf();

  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* mask_blob = BnInOp2Blob("idx");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  CHECK((in_diff_blob->shape().NumAxes() == 4)
        && (out_diff_blob->shape().NumAxes() == 4));

  PoolingKernelUtil<device_type, FloatingPointType>::PoolingBackward(
      ctx, out_diff_blob, mask_blob, in_diff_blob, pooling_conf);
}

template<typename FloatingPointType>
class PoolingKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx& ctx, const Blob* in_blob,
                             Blob* out_blob, Blob* mask_blob,
                             const PoolingOpConf& pooling_conf) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      const FloatingPointType* in_dptr = in_blob->dptr<FloatingPointType>();
      FloatingPointType* out_dptr = out_blob->mut_dptr<FloatingPointType>();
      int64_t* mask_dptr = mask_blob->mut_dptr<int64_t>();

      switch (pooling_conf.pool()) {
        case PoolingOpConf::MAX: {
          for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_blob->shape().At(3);
                     ++out_w) {
                  int64_t hstart =
                      out_h * pooling_conf.stride(0) - pooling_conf.pad(0);
                  int64_t wstart =
                      out_w * pooling_conf.stride(1) - pooling_conf.pad(1);
                  int64_t hend = std::min(hstart + pooling_conf.kernel_size(0),
                                          in_blob->shape().At(2));
                  int64_t wend = std::min(wstart + pooling_conf.kernel_size(1),
                                          in_blob->shape().At(3));
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  const int64_t out_index =
                      out_h * out_blob->shape().At(3) + out_w;
                  out_dptr[out_index] =
                      in_dptr[hstart * in_blob->shape().At(3) + wstart];
                  mask_dptr[out_index] =
                      hstart * in_blob->shape().At(3) + wstart;
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      const int64_t index = h * in_blob->shape().At(3) + w;
                      if (in_dptr[index] > out_dptr[out_index]) {
                        out_dptr[out_index] = in_dptr[index];
                        mask_dptr[out_index] = index;
                      }
                    }
                  }
                }
              }
              in_dptr += in_blob->shape().Count(2);
              out_dptr += out_blob->shape().Count(2);
              mask_dptr += out_blob->shape().Count(2);
            }
          }
          break;
        }
        case PoolingOpConf::AVE: {
          for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_blob->shape().At(3);
                     ++out_w) {
                  int64_t hstart =
                      out_h * pooling_conf.stride(0) - pooling_conf.pad(0);
                  int64_t wstart =
                      out_w * pooling_conf.stride(1) - pooling_conf.pad(1);
                  int64_t hend =
                      std::min(hstart + pooling_conf.kernel_size(0),
                               in_blob->shape().At(2) + pooling_conf.pad(0));
                  int64_t wend =
                      std::min(wstart + pooling_conf.kernel_size(1),
                               in_blob->shape().At(3) + pooling_conf.pad(1));
                  int64_t pool_size = (hend - hstart) * (wend - wstart);
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  hend = std::min(hend, in_blob->shape().At(2));
                  wend = std::min(wend, in_blob->shape().At(3));
                  const int64_t out_index =
                      out_h * out_blob->shape().At(3) + out_w;
                  out_dptr[out_index] = 0;
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      out_dptr[out_index] +=
                          in_dptr[h * in_blob->shape().At(3) + w];
                    }
                  }
                  out_dptr[out_index] /= pool_size;
                }
              }
              in_dptr += in_blob->shape().Count(2);
              out_dptr += out_blob->shape().Count(2);
            }
          }
          break;
        }
        case PoolingOpConf::STOCHASTIC: {
          TODO();
        }
        /*
        for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
          for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
            for (int64_t out_h = 0; out_h < out_blob->shape().At(2); ++out_h) {
              for (int64_t out_w = 0; out_w < out_blob->shape().At(3); ++out_w)
        { int64_t hstart = out_h * pooling_conf.stride(0) - pooling_conf.pad(0);
                int64_t wstart = out_w * pooling_conf.stride(1) -
        pooling_conf.pad(1); int64_t hend = std::min(hstart +
        pooling_conf.kernel_size(0), in_blob->shape().At(2)); int64_t wend =
                    std::min(wstart + pooling_conf.kernel_size(1),
        in_blob->shape().At(3)); hstart = std::max(hstart,
        static_cast<int64_t>(0)); wstart = std::max(wstart,
        static_cast<int64_t>(0)); const int64_t out_index = out_h *
        out_blob->shape().At(3) + out_w; std::mt19937
        generator(NewRandomSeed());
                const int64_t index = (hstart + (generator() % (hend - hstart)))
        * in_width + (wstart + (generator() % (wend - wstart)));
                out_dptr[out_index] = in_dptr[index];
                mask_dptr[out_index] = index;
              }
            }
            in_dptr += in_blob->shape().Count(2);
            out_dptr += out_blob->shape().Count(2);
          }
        }
        break;
        */
        default: { UNEXPECTED_RUN(); }
      }
    });
  }

  static void PoolingBackward(const KernelCtx& ctx, const Blob* out_diff_blob,
                              const Blob* mask_blob, Blob* in_diff_blob,
                              const PoolingOpConf& pooling_conf) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      const FloatingPointType* out_diff_dptr =
          out_diff_blob->dptr<FloatingPointType>();
      const int64_t* mask_dptr = mask_blob->dptr<int64_t>();
      FloatingPointType* in_diff_dptr =
          in_diff_blob->mut_dptr<FloatingPointType>();
      memset(in_diff_dptr, 0,
             in_diff_blob->shape().elem_cnt() * sizeof(FloatingPointType));
      switch (pooling_conf.pool()) {
        case PoolingOpConf::MAX: {
          for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3);
                     ++out_w) {
                  const int64_t out_diff_index =
                      out_h * out_diff_blob->shape().At(3) + out_w;
                  const int64_t in_diff_index = mask_dptr[out_diff_index];
                  in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
                }
              }
              out_diff_dptr += out_diff_blob->shape().Count(2);
              mask_dptr += out_diff_blob->shape().Count(2);
              in_diff_dptr += in_diff_blob->shape().Count(2);
            }
          }
          break;
        }
        case PoolingOpConf::AVE: {
          for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3);
                     ++out_w) {
                  int64_t hstart =
                      out_h * pooling_conf.stride(0) - pooling_conf.pad(0);
                  int64_t wstart =
                      out_w * pooling_conf.stride(1) - pooling_conf.pad(1);
                  int64_t hend = std::min(
                      hstart + pooling_conf.kernel_size(0),
                      in_diff_blob->shape().At(2) + pooling_conf.pad(0));
                  int64_t wend = std::min(
                      wstart + pooling_conf.kernel_size(1),
                      in_diff_blob->shape().At(3) + pooling_conf.pad(1));
                  int64_t pool_size = (hend - hstart) * (wend - wstart);
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  hend = std::min(hend, in_diff_blob->shape().At(2));
                  wend = std::min(wend, in_diff_blob->shape().At(3));
                  const int64_t out_diff_index =
                      out_h * out_diff_blob->shape().At(3) + out_w;
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      const int64_t in_diff_index =
                          h * in_diff_blob->shape().At(3) + w;
                      in_diff_dptr[in_diff_index] +=
                          out_diff_dptr[out_diff_index] / pool_size;
                    }
                  }
                }
              }
              out_diff_dptr += out_diff_blob->shape().Count(2);
              in_diff_dptr += in_diff_blob->shape().Count(2);
            }
          }
          break;
        }
        case PoolingOpConf::STOCHASTIC: {
          TODO();
        }
        /*
        for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
          for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
            for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2);
        ++out_h) { for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3);
        ++out_w) { const int out_diff_index = out_h * out_blob->shape().At(3) +
        out_w; const int in_diff_index = mask[out_diff_index];
                in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
              }
            }
            out_diff_dptr += out_diff_blob->shape().Count(2);
            mask_dptr += out_diff_blob->shape().Count(2);
            in_diff_dptr += in_diff_blob->shape().Count(2);
          }
        }
        break;
        */
        default: { UNEXPECTED_RUN(); }
      }
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(PoolingKernelUtil);
INSTANTIATE_KERNEL_CLASS(PoolingKernel);
REGISTER_KERNEL(OperatorConf::kPoolingConf, PoolingKernel);

}  // namespace oneflow
