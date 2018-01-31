#include "oneflow/core/kernel/average_pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void AveragePoolingKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  AveragePoolingKernelUtil<device_type, T>::Forward(ctx, in_blob, out_blob,
                                                    this->pooling_ctx());
}

template<DeviceType device_type, typename T>
void AveragePoolingKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  AveragePoolingKernelUtil<device_type, T>::Backward(
      ctx, out_diff_blob, in_diff_blob, this->pooling_ctx());
}

template<DeviceType device_type, typename T>
const PoolingKernelConf&
AveragePoolingKernel<device_type, T>::GetPoolingKernelConf() const {
  return this->kernel_conf().average_pooling_conf().pooling_conf();
}

template<typename T>
class AveragePoolingKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingKernelUtil);
  AveragePoolingKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const Blob* in_blob, Blob* out_blob,
                      const PoolingCtx& pooling_ctx) {
    const T* in_dptr = in_blob->dptr<T>();
    T* out_dptr = out_blob->mut_dptr<T>();

    FOR_RANGE(int64_t, n, 0, out_blob->shape().At(0)) {
      FOR_RANGE(int64_t, c, 0, out_blob->shape().At(1)) {
        FOR_RANGE(int64_t, out_h, 0, out_blob->shape().At(2)) {
          FOR_RANGE(int64_t, out_w, 0, out_blob->shape().At(3)) {
            int64_t hstart =
                out_h * pooling_ctx.strides_h - pooling_ctx.padding_top;
            int64_t wstart =
                out_w * pooling_ctx.strides_w - pooling_ctx.padding_left;
            int64_t hend =
                std::min(hstart + pooling_ctx.pool_size_h,
                         in_blob->shape().At(2) + pooling_ctx.padding_bottom);
            int64_t wend =
                std::min(wstart + pooling_ctx.pool_size_w,
                         in_blob->shape().At(3) + pooling_ctx.padding_right);
            int64_t pool_size = (hend - hstart) * (wend - wstart);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            wstart = std::max(wstart, static_cast<int64_t>(0));
            hend = std::min(hend, in_blob->shape().At(2));
            wend = std::min(wend, in_blob->shape().At(3));
            const int64_t out_index = out_h * out_blob->shape().At(3) + out_w;
            out_dptr[out_index] = 0;
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                out_dptr[out_index] += in_dptr[h * in_blob->shape().At(3) + w];
              }
            }
            out_dptr[out_index] /= pool_size;
          }
        }
        in_dptr += in_blob->shape().Count(2);
        out_dptr += out_blob->shape().Count(2);
      }
    }
  }

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       Blob* in_diff_blob, const PoolingCtx& pooling_ctx) {
    const T* out_diff_dptr = out_diff_blob->dptr<T>();
    T* in_diff_dptr = in_diff_blob->mut_dptr<T>();

    FOR_RANGE(int64_t, n, 0, out_diff_blob->shape().At(0)) {
      FOR_RANGE(int64_t, c, 0, out_diff_blob->shape().At(1)) {
        FOR_RANGE(int64_t, out_h, 0, out_diff_blob->shape().At(2)) {
          FOR_RANGE(int64_t, out_w, 0, out_diff_blob->shape().At(3)) {
            int64_t hstart =
                out_h * pooling_ctx.strides_h - pooling_ctx.padding_top;
            int64_t wstart =
                out_w * pooling_ctx.strides_w - pooling_ctx.padding_left;
            int64_t hend = std::min(
                hstart + pooling_ctx.pool_size_h,
                in_diff_blob->shape().At(2) + pooling_ctx.padding_bottom);
            int64_t wend = std::min(
                wstart + pooling_ctx.pool_size_w,
                in_diff_blob->shape().At(3) + pooling_ctx.padding_right);
            int64_t pool_size = (hend - hstart) * (wend - wstart);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            wstart = std::max(wstart, static_cast<int64_t>(0));
            hend = std::min(hend, in_diff_blob->shape().At(2));
            wend = std::min(wend, in_diff_blob->shape().At(3));
            const int64_t out_diff_index =
                out_h * out_diff_blob->shape().At(3) + out_w;
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
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
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePoolingConf,
                           AveragePoolingKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
