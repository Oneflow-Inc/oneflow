#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
PoolingKernel<device_type, T>::PoolingKernel() {
#ifdef USE_CUDNN
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreatePoolingDescriptor(&pooling_desc_));
#endif
}

template<DeviceType device_type, typename T>
PoolingKernel<device_type, T>::~PoolingKernel() {
#ifdef USE_CUDNN
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyPoolingDescriptor(pooling_desc_));
#endif
}

template<DeviceType device_type, typename T>
void PoolingKernel<device_type, T>::InitFromOpProto(
    const OperatorProto& op_proto) {
#ifdef USE_CUDNN
  Kernel::InitFromOpProto(op_proto);

  const auto pooling_conf = op()->op_conf().pooling_conf();

  switch (pooling_conf.pool()) {
    case PoolingOpConf::kMax: {
      pooling_mode_ = CUDNN_POOLING_MAX;
      break;
    }
    case PoolingOpConf::kAve: {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    }
    default: { UNEXPECTED_RUN(); }
  }

  CudaCheck(cudnnSetPooling2dDescriptor(
      pooling_desc_, pooling_mode_, CUDNN_PROPAGATE_NAN,
      pooling_conf.kernel_h(), pooling_conf.kernel_w(), pooling_conf.pad_h(),
      pooling_conf.pad_w(), pooling_conf.stride_h(), pooling_conf.stride_w()));
#endif
}

template<DeviceType device_type, typename T>
void PoolingKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PoolingOpConf& pooling_conf = op()->op_conf().pooling_conf();

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CopyDataIdFromSoleIbToAllObIfNeed<device_type>(ctx,
                                                 BnInOp2Blob);  // TODO(shiyuan)

#ifdef USE_CUDNN
  CudaCheck(cudnnSetTensor4dDescriptor(
      in_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      in_blob->shape().At(0), in_blob->shape().At(1), in_blob->shape().At(2),
      in_blob->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      out_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      out_blob->shape().At(0), out_blob->shape().At(1), out_blob->shape().At(3),
      out_blob->shape().At(3)));

  CudaCheck(cudnnPoolingForward(ctx.device_ctx->cudnn_handle(), pooling_desc_,
                                cudnn::DataType<T>::one, in_desc_,
                                in_blob->dptr<T>(), cudnn::DataType<T>::zero,
                                out_desc_, out_blob->mut_dptr<T>()));
#else
  Blob* idx_blob = BnInOp2Blob("idx");
  PoolingKernelUtil<device_type, T>::PoolingForward(ctx, in_blob, out_blob,
                                                    idx_blob, pooling_conf);
#endif
}

template<DeviceType device_type, typename T>
void PoolingKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataField());  // TODO(shiyuan)
  const PoolingOpConf& pooling_conf = op()->op_conf().pooling_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

#ifdef USE_CUDNN
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  CudaCheck(cudnnPoolingBackward(
      ctx.device_ctx->cudnn_handle(), pooling_desc_, cudnn::DataType<T>::one,
      out_desc_, out_blob->dptr<T>(), out_desc_, out_diff_blob->dptr<T>(),
      in_desc_, in_blob->dptr<T>(), cudnn::DataType<T>::zero, in_desc_,
      in_diff_blob->mut_dptr<T>()));
#else
  const Blob* idx_blob = BnInOp2Blob("idx");
  PoolingKernelUtil<device_type, T>::PoolingBackward(
      ctx, out_diff_blob, idx_blob, in_diff_blob, pooling_conf);
#endif
}

template<typename T>
class PoolingKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx& ctx, const Blob* in_blob,
                             Blob* out_blob, Blob* idx_blob,
                             const PoolingOpConf& pooling_conf) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      const T* in_dptr = in_blob->dptr<T>();
      T* out_dptr = out_blob->mut_dptr<T>();
      uint32_t* idx_dptr = idx_blob->mut_dptr<uint32_t>();

      switch (pooling_conf.pool()) {
        case PoolingOpConf::kMax: {
          for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_blob->shape().At(3);
                     ++out_w) {
                  int64_t hstart =
                      out_h * pooling_conf.stride_h() - pooling_conf.pad_h();
                  int64_t wstart =
                      out_w * pooling_conf.stride_w() - pooling_conf.pad_w();
                  int64_t hend = std::min(hstart + pooling_conf.kernel_h(),
                                          in_blob->shape().At(2));
                  int64_t wend = std::min(wstart + pooling_conf.kernel_w(),
                                          in_blob->shape().At(3));
                  hstart = std::max(hstart, static_cast<int64_t>(0));
                  wstart = std::max(wstart, static_cast<int64_t>(0));
                  const int64_t out_index =
                      out_h * out_blob->shape().At(3) + out_w;
                  out_dptr[out_index] =
                      in_dptr[hstart * in_blob->shape().At(3) + wstart];
                  idx_dptr[out_index] =
                      hstart * in_blob->shape().At(3) + wstart;
                  for (int64_t h = hstart; h < hend; ++h) {
                    for (int64_t w = wstart; w < wend; ++w) {
                      const uint32_t index = h * in_blob->shape().At(3) + w;
                      if (in_dptr[index] > out_dptr[out_index]) {
                        out_dptr[out_index] = in_dptr[index];
                        idx_dptr[out_index] = index;
                      }
                    }
                  }
                }
              }
              in_dptr += in_blob->shape().Count(2);
              out_dptr += out_blob->shape().Count(2);
              idx_dptr += out_blob->shape().Count(2);
            }
          }
          break;
        }
        case PoolingOpConf::kAve: {
          for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_blob->shape().At(3);
                     ++out_w) {
                  int64_t hstart =
                      out_h * pooling_conf.stride_h() - pooling_conf.pad_h();
                  int64_t wstart =
                      out_w * pooling_conf.stride_w() - pooling_conf.pad_w();
                  int64_t hend =
                      std::min(hstart + pooling_conf.kernel_h(),
                               in_blob->shape().At(2) + pooling_conf.pad_h());
                  int64_t wend =
                      std::min(wstart + pooling_conf.kernel_w(),
                               in_blob->shape().At(3) + pooling_conf.pad_w());
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
        case PoolingOpConf::kStochastic: {
          TODO();
        }
        /*
        for (int64_t n = 0; n < out_blob->shape().At(0); ++n) {
          for (int64_t c = 0; c < out_blob->shape().At(1); ++c) {
            for (int64_t out_h = 0; out_h < out_blob->shape().At(2); ++out_h)
            {
              for (int64_t out_w = 0; out_w < out_blob->shape().At(3);
              ++out_w)
        { int64_t hstart = out_h * pooling_conf.stride_h() -
        pooling_conf.pad_h(); int64_t wstart = out_w * pooling_conf.stride_w()
        - pooling_conf.pad_w(); int64_t hend = std::min(hstart +
        pooling_conf.kernel_h(), in_blob->shape().At(2)); int64_t wend =
                    std::min(wstart + pooling_conf.kernel_w(),
        in_blob->shape().At(3)); hstart = std::max(hstart,
        static_cast<int64_t>(0)); wstart = std::max(wstart,
        static_cast<int64_t>(0)); const int64_t out_index = out_h *
        out_blob->shape().At(3) + out_w; std::mt19937
        generator(NewRandomSeed());
                const int64_t index = (hstart + (generator() % (hend -
                hstart)))
        * in_width + (wstart + (generator() % (wend - wstart)));
                out_dptr[out_index] = in_dptr[index];
                idx_dptr[out_index] = index;
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
                              const Blob* idx_blob, Blob* in_diff_blob,
                              const PoolingOpConf& pooling_conf) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      const T* out_diff_dptr = out_diff_blob->dptr<T>();
      const uint32_t* idx_dptr = idx_blob->dptr<uint32_t>();
      T* in_diff_dptr = in_diff_blob->mut_dptr<T>();
      switch (pooling_conf.pool()) {
        case PoolingOpConf::kMax: {
          for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3);
                     ++out_w) {
                  const int64_t out_diff_index =
                      out_h * out_diff_blob->shape().At(3) + out_w;
                  const uint32_t in_diff_index = idx_dptr[out_diff_index];
                  in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
                }
              }
              out_diff_dptr += out_diff_blob->shape().Count(2);
              idx_dptr += out_diff_blob->shape().Count(2);
              in_diff_dptr += in_diff_blob->shape().Count(2);
            }
          }
          break;
        }
        case PoolingOpConf::kAve: {
          for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
            for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
              for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2);
                   ++out_h) {
                for (int64_t out_w = 0; out_w < out_diff_blob->shape().At(3);
                     ++out_w) {
                  int64_t hstart =
                      out_h * pooling_conf.stride_h() - pooling_conf.pad_h();
                  int64_t wstart =
                      out_w * pooling_conf.stride_w() - pooling_conf.pad_w();
                  int64_t hend = std::min(
                      hstart + pooling_conf.kernel_h(),
                      in_diff_blob->shape().At(2) + pooling_conf.pad_h());
                  int64_t wend = std::min(
                      wstart + pooling_conf.kernel_w(),
                      in_diff_blob->shape().At(3) + pooling_conf.pad_w());
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
        case PoolingOpConf::kStochastic: {
          TODO();
        }
        /*
        for (int64_t n = 0; n < out_diff_blob->shape().At(0); ++n) {
          for (int64_t c = 0; c < out_diff_blob->shape().At(1); ++c) {
            for (int64_t out_h = 0; out_h < out_diff_blob->shape().At(2);
        ++out_h) { for (int64_t out_w = 0; out_w <
        out_diff_blob->shape().At(3);
        ++out_w) { const int out_diff_index = out_h * out_blob->shape().At(3)
        + out_w; const int in_diff_index = idx[out_diff_index];
                in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
              }
            }
            out_diff_dptr += out_diff_blob->shape().Count(2);
            idx_dptr += out_diff_blob->shape().Count(2);
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

namespace {

Kernel* CreatePoolingKenrel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define POOLING_KERNEL_ENTRY(device_type, data_type_pair)             \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() { \
     return new PoolingKernel<device_type,                            \
                              OF_PP_PAIR_FIRST(data_type_pair)>();    \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(POOLING_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(op_ctx.device_type(), op_ctx.bn_in_op2data_type().at("in")))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kPoolingConf, CreatePoolingKenrel))

}  // namespace oneflow
