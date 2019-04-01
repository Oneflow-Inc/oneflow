#include "oneflow/core/kernel/pooling_grad_kernel.h"

namespace oneflow {

template<typename T>
void PoolingGradKernel<DeviceType::kCPU, T>::PoolingBackward(const KernelCtx& kernel_ctx,
                                                             const PoolingCtx& pooling_ctx,
                                                             const Blob* dy_blob,
                                                             const Blob* y_blob, const Blob* x_blob,
                                                             Blob* dx_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->BackwardNCDHW(pooling_ctx, dy_blob, y_blob, x_blob, dx_blob);
  } else if (data_format == "channels_last") {
    this->BackwardNDHWC(pooling_ctx, dy_blob, y_blob, x_blob, dx_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void PoolingGradKernel<DeviceType::kCPU, T>::BackwardNCDHW(const PoolingCtx& ctx,
                                                           const Blob* dy_blob, const Blob* y_blob,
                                                           const Blob* x_blob,
                                                           Blob* dx_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  const T* dy = dy_blob->dptr<T>();
  const T* output = y_blob->dptr<T>();
  const T* input = x_blob->dptr<T>();
  T* dx = dx_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.Get(0) - padding_before.Get(0);
        int64_t dend = std::min(dstart + pool_size.Get(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.Get(1) - padding_before.Get(1);
          int64_t hend = std::min(hstart + pool_size.Get(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.Get(2) - padding_before.Get(2);
            int64_t wend = std::min(wstart + pool_size.Get(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t index = d * in.Count(3) + h * in.At(4) + w;
                  NCDHWProcessGrad(input[index], output[pool_index], dy[pool_index], size,
                                   dx[index]);
                }
              }
            }
          }
        }
      }
      // offset
      input += in.Count(2);
      dx += in.Count(2);
      output += out.Count(2);
      dy += out.Count(2);
    }
  }
}

template<typename T>
void PoolingGradKernel<DeviceType::kCPU, T>::BackwardNDHWC(const PoolingCtx& ctx,
                                                           const Blob* dy_blob, const Blob* y_blob,
                                                           const Blob* x_blob,
                                                           Blob* dx_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  // caffe2 implementation: need check
  ConstEigenArrayMap<T> out_mat(y_blob->dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  ConstEigenArrayMap<T> in_mat(x_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  ConstEigenArrayMap<T> out_diff_mat(dy_blob->dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  EigenArrayMap<T> in_diff_mat(dx_blob->mut_dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(2)) {
      int64_t dstart = pd * strides.Get(0) - padding_before.Get(0);
      int64_t dend = std::min(dstart + pool_size.Get(0), in.At(2));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(3)) {
        int64_t hstart = ph * strides.Get(1) - padding_before.Get(1);
        int64_t hend = std::min(hstart + pool_size.Get(1), in.At(3));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(4)) {
          int64_t wstart = pw * strides.Get(2) - padding_before.Get(2);
          int64_t wend = std::min(wstart + pool_size.Get(2), in.At(4));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int64_t pool_index = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
          const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int64_t input_index = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                NDHWCProcessGrad(pool_index, input_index, size, out_mat, in_mat, out_diff_mat,
                                 in_diff_mat);
              }
            }
          }
        }
      }
    }
  }
}

#define INSTANTIATE_POOLING_GRAD_KERNEL(type_cpp, type_proto) \
  template class PoolingGradKernel<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_GRAD_KERNEL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
