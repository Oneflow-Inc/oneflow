#include "oneflow/core/kernel/max_pooling_2d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MaxPooling2DKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* idx_blob = BnInOp2Blob("idx");
  MaxPooling2DKernelUtil<device_type, T>::Forward(
      ctx, in_blob, out_blob, idx_blob, this->pooling_2d_ctx());
}

template<DeviceType device_type, typename T>
void MaxPooling2DKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* idx_blob = BnInOp2Blob("idx");
  MaxPooling2DKernelUtil<device_type, T>::Backward(
      ctx, out_diff_blob, idx_blob, in_diff_blob, this->pooling_2d_ctx());
}

template<DeviceType device_type, typename T>
const Pooling2DKernelConf&
MaxPooling2DKernel<device_type, T>::GetPooling2DKernelConf() const {
  return this->kernel_conf().max_pooling_2d_conf().pooling_2d_conf();
}

template<DeviceType device_type, typename T>
const PbMessage& MaxPooling2DKernel<device_type, T>::GetPooling2DOpConf()
    const {
  return this->op_conf().max_pooling_2d_conf();
}

template<typename T>
class MaxPooling2DKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling2DKernelUtil);
  MaxPooling2DKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const Blob* in_blob, Blob* out_blob,
                      Blob* idx_blob, const Pooling2DCtx& pooling_ctx) {
    const T* in_dptr = in_blob->dptr<T>();
    T* out_dptr = out_blob->mut_dptr<T>();
    int32_t* idx_dptr = idx_blob->mut_dptr<int32_t>();

    FOR_RANGE(int64_t, n, 0, out_blob->shape().At(0)) {
      FOR_RANGE(int64_t, c, 0, out_blob->shape().At(1)) {
        FOR_RANGE(int64_t, out_h, 0, out_blob->shape().At(2)) {
          FOR_RANGE(int64_t, out_w, 0, out_blob->shape().At(3)) {
            int64_t hstart =
                out_h * pooling_ctx.strides_h - pooling_ctx.padding_top;
            int64_t wstart =
                out_w * pooling_ctx.strides_w - pooling_ctx.padding_left;
            int64_t hend = std::min(hstart + pooling_ctx.pool_size_h,
                                    in_blob->shape().At(2));
            int64_t wend = std::min(wstart + pooling_ctx.pool_size_w,
                                    in_blob->shape().At(3));
            hstart = std::max(hstart, static_cast<int64_t>(0));
            wstart = std::max(wstart, static_cast<int64_t>(0));
            const int64_t out_index = out_h * out_blob->shape().At(3) + out_w;
            out_dptr[out_index] =
                in_dptr[hstart * in_blob->shape().At(3) + wstart];
            idx_dptr[out_index] = hstart * in_blob->shape().At(3) + wstart;
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int32_t index = h * in_blob->shape().At(3) + w;
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
  }

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       const Blob* idx_blob, Blob* in_diff_blob,
                       const Pooling2DCtx& pooling_ctx) {
    const T* out_diff_dptr = out_diff_blob->dptr<T>();
    const int32_t* idx_dptr = idx_blob->dptr<int32_t>();
    T* in_diff_dptr = in_diff_blob->mut_dptr<T>();

    FOR_RANGE(int64_t, n, 0, out_diff_blob->shape().At(0)) {
      FOR_RANGE(int64_t, c, 0, out_diff_blob->shape().At(1)) {
        FOR_RANGE(int64_t, out_h, 0, out_diff_blob->shape().At(2)) {
          FOR_RANGE(int64_t, out_w, 0, out_diff_blob->shape().At(3)) {
            const int64_t out_diff_index =
                out_h * out_diff_blob->shape().At(3) + out_w;
            const int32_t in_diff_index = idx_dptr[out_diff_index];
            in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
          }
        }
        out_diff_dptr += out_diff_blob->shape().Count(2);
        idx_dptr += out_diff_blob->shape().Count(2);
        in_diff_dptr += in_diff_blob->shape().Count(2);
      }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling2DConf, MaxPooling2DKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
