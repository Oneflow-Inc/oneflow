#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA
CudnnPoolingDesc::~CudnnPoolingDesc() { CudaCheck(cudnnDestroyPoolingDescriptor(val_)); }

CudnnPoolingDesc::CudnnPoolingDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window,
                                   const int* padding, const int* stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, dims, window,
                                        padding, stride));
}
#endif

PoolingCtx::PoolingCtx(const PoolingKernelConf& kernel_conf
#ifdef WITH_CUDA
                       ,
                       cudnnPoolingMode_t pooling_mode, DataType type
#endif  // WITH_CUDA
                       )
    : kernel_conf_(kernel_conf), pooling_mode_(pooling_mode) {
#ifdef WITH_CUDA
  int32_t dim = kernel_conf_.dim();
  CHECK_GE(dim, 1);
  CHECK_LE(dim, 3);
  std::vector<int> in_dim = GetStdVecFromShapeInKernelConf("in");
  std::vector<int> out_dim = GetStdVecFromShapeInKernelConf("out");

  std::vector<int> pool_size(dim);
  std::vector<int> padding(dim);
  std::vector<int> strides(dim);
  FOR_RANGE(int, i, 0, dim) {
    int32_t index_in_3d = i + 3 - dim;
    pool_size[i] = kernel_conf_.pool_size().Get(index_in_3d);
    padding[i] = std::max(kernel_conf_.padding_before().Get(index_in_3d),
                          kernel_conf_.padding_after().Get(index_in_3d));
    strides[i] = kernel_conf_.strides().Get(index_in_3d);
  }
  pooling_desc_.reset(
      new CudnnPoolingDesc(pooling_mode_, dim, pool_size.data(), padding.data(), strides.data()));

  int32_t ncx_dim = 2 + dim;
  std::vector<int> in_shape(ncx_dim);
  std::vector<int> out_shape(ncx_dim);
  std::vector<int> in_stride(ncx_dim);
  std::vector<int> out_stride(ncx_dim);

  FOR_RANGE(size_t, i, 0, 2) {
    in_shape[i] = in_dim[i];
    out_shape[i] = out_dim[i];
  }
  FOR_RANGE(int, i, 0, dim) {
    int32_t index_in_3d = 2 + i + 3 - dim;
    in_shape[i + 2] = in_dim[index_in_3d];
    out_shape[i + 2] = out_dim[index_in_3d];
  }

  const std::string& data_format = kernel_conf_.data_format();
  if (data_format == "channels_first") {
    in_stride[ncx_dim - 1] = 1;
    out_stride[ncx_dim - 1] = 1;

    for (int i = ncx_dim - 2; i >= 0; --i) {
      in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
      out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
  } else if (data_format == "channels_last") {
    in_stride[ncx_dim - 1] = in_shape[1];
    out_stride[ncx_dim - 1] = out_shape[1];
    for (int i = ncx_dim - 2; i >= 2; --i) {
      in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
      out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
    in_stride[1] = 1;
    out_stride[1] = 1;
    in_stride[0] = in_shape[2] * in_stride[2];
    out_stride[0] = out_shape[2] * out_stride[2];
  } else {
    UNIMPLEMENTED();
  }
  in_desc_.reset(new CudnnTensorDesc(type, ncx_dim, in_shape.data(), in_stride.data()));
  out_desc_.reset(new CudnnTensorDesc(type, ncx_dim, out_shape.data(), out_stride.data()));
#endif  // WITH_CUDA
}

#ifdef WITH_CUDA
const cudnnTensorDescriptor_t& PoolingCtx::cudnn_in_tensor_desc() const { return in_desc_->Get(); }

const cudnnTensorDescriptor_t& PoolingCtx::cudnn_out_tensor_desc() const {
  return out_desc_->Get();
}

const cudnnPoolingDescriptor_t& PoolingCtx::cudnn_pooling_desc() const {
  return pooling_desc_->Get();
}

#endif  // WITH_CUDA

PoolingKernelConf GenPoolingKernelConfForNewInShape(const Shape& in_shape,
                                                    const PoolingKernelConf& conf) {
  PoolingKernelConf ret = conf;
#ifdef WITH_CUDA
  int32_t dim = conf.dim();
  std::string data_format = conf.data_format();
  std::vector<int64_t> in = {GetInDim(in_shape, data_format, 0, dim),
                             GetInDim(in_shape, data_format, 1, dim),
                             GetInDim(in_shape, data_format, 2, dim)};
  std::vector<int64_t> out;
  std::vector<int32_t> padding_before;
  std::vector<int32_t> padding_after;
  Get3DOutputSize(in, PbRf2StdVec(conf.pool_size()), PbRf2StdVec(conf.strides()),
                  conf.padding_type(), &out, &padding_before, &padding_after);

  FOR_RANGE(size_t, i, 0, 3) {
    ret.mutable_in()->set_dim(i + 2, in.at(i));
    ret.mutable_out()->set_dim(i + 2, out.at(i));
    ret.set_padding_before(i, padding_before.at(i));
    ret.set_padding_after(i, padding_after.at(i));
  }
#endif  // WITH_CUDA
  return ret;
}

std::vector<int> PoolingCtx::GetStdVecFromShapeInKernelConf(const std::string& field_name) const {
  const PbRf<int64_t>& shape = GetPbRfFromPbMessage<int64_t>(
      GetValFromPbMessage<const PbMessage&>(kernel_conf_, field_name), "dim");
  std::vector<int> ret(shape.begin(), shape.end());
  return ret;
}

template<typename T>
void PoolingKernel<DeviceType::kCPU, T>::PoolingForward(const KernelCtx& kernel_ctx,
                                                        const PoolingCtx& pooling_ctx,
                                                        const Blob* in_blob, Blob* out_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->ForwardNCDHW(pooling_ctx, in_blob, out_blob);
  } else if (data_format == "channels_last") {
    this->ForwardNDHWC(pooling_ctx, in_blob, out_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void PoolingKernel<DeviceType::kCPU, T>::PoolingBackward(const KernelCtx& kernel_ctx,
                                                         const PoolingCtx& pooling_ctx,
                                                         const Blob* out_diff_blob,
                                                         const Blob* out_blob, const Blob* in_blob,
                                                         Blob* in_diff_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->BackwardNCDHW(pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
  } else if (data_format == "channels_last") {
    this->BackwardNDHWC(pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void PoolingKernel<DeviceType::kCPU, T>::ForwardNCDHW(const PoolingCtx& ctx, const Blob* in_blob,
                                                      Blob* out_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  const T* input = in_blob->dptr<T>();
  T* output = out_blob->mut_dptr<T>();
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

            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            T res = ForwardInitialize();
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t input_index = d * in.Count(3) + h * in.At(4) + w;
                  NCDHWProcess(input[input_index], res);
                }
              }
            }
            NCDHWFinalize((dend - dstart) * (hend - hstart) * (wend - wstart), res);
            output[pool_index] = res;
          }
        }
      }
      input += in.Count(2);
      output += out.Count(2);
    }
  }
}

template<typename T>
void PoolingKernel<DeviceType::kCPU, T>::BackwardNCDHW(const PoolingCtx& ctx,
                                                       const Blob* out_diff_blob,
                                                       const Blob* out_blob, const Blob* in_blob,
                                                       Blob* in_diff_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  const T* output_diff = out_diff_blob->dptr<T>();
  const T* output = out_blob->dptr<T>();
  const T* input = in_blob->dptr<T>();
  T* input_diff = in_diff_blob->mut_dptr<T>();
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
                  NCDHWProcessGrad(input[index], output[pool_index], output_diff[pool_index], size,
                                   input_diff[index]);
                }
              }
            }
          }
        }
      }
      // offset
      input += in.Count(2);
      input_diff += in.Count(2);
      output += out.Count(2);
      output_diff += out.Count(2);
    }
  }
}

template<typename T>
void PoolingKernel<DeviceType::kCPU, T>::ForwardNDHWC(const PoolingCtx& ctx, const Blob* in_blob,
                                                      Blob* out_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  EigenMatrixMap<T> out_mat(out_blob->mut_dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
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
          const int out_col = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
          out_mat.col(out_col).setConstant(ForwardInitialize());
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int in_col = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                NDHWCProcess(in_col, out_col, in_mat, out_mat);
              }
            }
          }
          NDHWCFinalize((hend - hstart) * (wend - wstart) * (dend - dstart), out_col, out_mat);
        }
      }
    }
  }
}

template<typename T>
void PoolingKernel<DeviceType::kCPU, T>::BackwardNDHWC(const PoolingCtx& ctx,
                                                       const Blob* out_diff_blob,
                                                       const Blob* out_blob, const Blob* in_blob,
                                                       Blob* in_diff_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  // caffe2 implementation: need check
  ConstEigenArrayMap<T> out_mat(out_blob->dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  ConstEigenArrayMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  ConstEigenArrayMap<T> out_diff_mat(out_diff_blob->dptr<T>(), out.At(1),
                                     out.elem_cnt() / out.At(1));
  EigenArrayMap<T> in_diff_mat(in_diff_blob->mut_dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
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

#define INSTANTIATE_POOLING_KERNEL(type_cpp, type_proto) \
  template class PoolingKernel<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
