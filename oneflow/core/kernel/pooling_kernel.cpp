#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA
CudnnPoolingNdDesc::~CudnnPoolingNdDesc() {
  CudaCheck(cudnnDestroyPoolingDescriptor(val_));
}

CudnnPoolingNdDesc::CudnnPoolingNdDesc(cudnnPoolingMode_t pooling_mode,
                                       const std::vector<int>& window,
                                       const std::vector<int>& padding,
                                       const std::vector<int>& stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(
      val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, window.size(), window.data(),
      padding.data(), stride.data()));
}
#endif

Pooling3DCtx::~Pooling3DCtx() {
#ifdef WITH_CUDA
  delete in_desc_;
  delete out_desc_;
  delete pooling_desc_;
#endif  // WITH_CUDA
}

void Pooling3DCtx::set_kernel_conf(const Pooling3DKernelConf& kernel_conf) {
  kernel_conf_ = kernel_conf;
}

#ifdef WITH_CUDA
void Pooling3DCtx::set_cudnn_pooling_mode(cudnnPoolingMode_t pooling_mode) {
  pooling_mode_ = pooling_mode;
}

void Pooling3DCtx::BuildCudnnDescs(DataType type) {
  std::vector<int> window = GetShapeInStdVec("pool_size");
  std::vector<int> padding_before = GetShapeInStdVec("padding_before");
  std::vector<int> padding_after = GetShapeInStdVec("padding_after");
  std::vector<int> padding;
  FOR_RANGE(size_t, i, 0, padding_before.size()) {
    padding.push_back(std::max(padding_before[i], padding_after[i]));
  }
  std::vector<int> stride = GetShapeInStdVec("strides");
  std::vector<int> in_dim = GetShapeInStdVec("in");
  std::vector<int> in_stride{in_dim[1] * in_dim[2] * in_dim[3] * in_dim[4],
                             in_dim[2] * in_dim[3] * in_dim[4],
                             in_dim[3] * in_dim[4], in_dim[4]};
  std::vector<int> out_dim = GetShapeInStdVec("out");
  std::vector<int> out_stride = {
      out_dim[1] * out_dim[2] * out_dim[3] * out_dim[4],
      out_dim[2] * out_dim[3] * out_dim[4], out_dim[3] * out_dim[4],
      out_dim[4]};

  const std::string& data_format = kernel_conf_.data_format();
  if (data_format == "channels_first") {
    in_stride.insert(in_stride.end(), 1);
    out_stride.insert(out_stride.end(), 1);
  } else if (data_format == "channels_last") {
    in_stride.insert(in_stride.begin() + 1, 1);
    out_stride.insert(out_stride.begin() + 1, 1);
  } else {
    UNEXPECTED_RUN();
  }
  pooling_desc_ =
      new CudnnPoolingNdDesc(pooling_mode_, window, padding, stride);
  in_desc_ = new CudnnTensorDesc(type, in_dim, in_stride);
  out_desc_ = new CudnnTensorDesc(type, out_dim, out_stride);
}
#endif  // WITH_CUDA

std::vector<int> Pooling3DCtx::GetShapeInStdVec(
    const std::string& field_name) const {
  PbRf<int64_t> shape = GetPbRfFromPbMessage<int64_t>(
      GetMessageFromPbMessage(kernel_conf_, field_name), "dim");
  std::vector<int> ret;
  FOR_RANGE(size_t, i, 0, shape.size()) { ret.push_back(shape.Get(i)); }
  return ret;
}

template<typename T>
void Pooling<DeviceType::kCPU, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  ForwardOnCPU(this->pooling_3d_ctx(), in_blob, out_blob);
}

template<typename T>
void Pooling<DeviceType::kCPU, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kCPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  BackwardOnCPU(this->pooling_3d_ctx(), out_diff_blob, out_blob, in_blob,
                in_diff_blob);
}

template<typename T>
void Pooling<DeviceType::kCPU, T>::ForwardOnCPUWithOrderNCDHW(
    const Pooling3DCtx& ctx, const Blob* in_blob, Blob* out_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  Shape pool_size(ctx.kernel_conf().pool_size());
  Shape strides(ctx.kernel_conf().strides());
  Shape padding_before(ctx.kernel_conf().padding_before());

  const T* input = in_blob->dptr<T>();
  T* output = out_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.At(0) - padding_before.At(0);
        int64_t dend = std::min(dstart + pool_size.At(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.At(1) - padding_before.At(1);
          int64_t hend = std::min(hstart + pool_size.At(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.At(2) - padding_before.At(2);
            int64_t wend = std::min(wstart + pool_size.At(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            T res = ForwardInitialize();
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t input_index =
                      d * in.Count(3) + h * in.At(4) + w;
                  ForwardProcess(input[input_index], res);
                }
              }
            }
            ForwardFinalize((dend - dstart) * (hend - hstart) * (wend - wstart),
                            res);
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
void Pooling<DeviceType::kCPU, T>::BackwardOnCPUWithOrderNCDHW(
    const Pooling3DCtx& ctx, const Blob* out_diff_blob, const Blob* out_blob,
    const Blob* in_blob, Blob* in_diff_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  Shape pool_size(ctx.kernel_conf().pool_size());
  Shape strides(ctx.kernel_conf().strides());
  Shape padding_before(ctx.kernel_conf().padding_before());

  const T* output_diff = out_diff_blob->dptr<T>();
  const T* output = out_blob->dptr<T>();
  const T* input = in_blob->dptr<T>();
  T* input_diff = in_diff_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.At(0) - padding_before.At(0);
        int64_t dend = std::min(dstart + pool_size.At(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.At(1) - padding_before.At(1);
          int64_t hend = std::min(hstart + pool_size.At(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.At(2) - padding_before.At(2);
            int64_t wend = std::min(wstart + pool_size.At(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            float scale =
                1. / (hend - hstart) / (wend - wstart) / (dend - dstart);
            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t index = d * in.Count(3) + h * in.At(4) + w;
                  BackwardProcessGrad(input[index], output[pool_index],
                                      output_diff[pool_index], scale,
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
void Pooling<DeviceType::kCPU, T>::ForwardOnCPUWithOrderNDHWC(
    const Pooling3DCtx& ctx, const Blob* in_blob, Blob* out_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  Shape pool_size(ctx.kernel_conf().pool_size());
  Shape strides(ctx.kernel_conf().strides());
  Shape padding_before(ctx.kernel_conf().padding_before());

  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> in_mat(
      in_blob->dptr<T>(), in.At(4), in.elem_cnt() / in.At(4));
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> out_mat(
      out_blob->mut_dptr<T>(), out.At(4), out.elem_cnt() / out.At(4));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(1)) {
      int64_t dstart = pd * strides.At(0) - padding_before.At(0);
      int64_t dend = std::min(dstart + pool_size.At(0), in.At(1));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(2)) {
        int64_t hstart = ph * strides.At(1) - padding_before.At(1);
        int64_t hend = std::min(hstart + pool_size.At(1), in.At(2));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(3)) {
          int64_t wstart = pw * strides.At(2) - padding_before.At(2);
          int64_t wend = std::min(wstart + pool_size.At(2), in.At(3));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int out_col =
              ((n * out.At(1) + pd) * out.At(2) + ph) * out.At(3) + pw;
          out_mat.col(out_col).setConstant(ForwardInitialize());
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int in_col =
                    ((n * in.At(1) + d) * in.At(2) + h) * in.At(3) + w;
                ForwardProcess(in_col, out_col, in_mat, out_mat);
              }
            }
          }
          ForwardFinalize((hend - hstart) * (wend - wstart) * (dend - dstart),
                          out_col, out_mat);
        }
      }
    }
  }
}

template<typename T>
void Pooling<DeviceType::kCPU, T>::BackwardOnCPUWithOrderNDHWC(
    const Pooling3DCtx& ctx, const Blob* out_diff_blob, const Blob* out_blob,
    const Blob* in_blob, Blob* in_diff_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  Shape pool_size(ctx.kernel_conf().pool_size());
  Shape strides(ctx.kernel_conf().strides());
  Shape padding_before(ctx.kernel_conf().padding_before());

  // caffe2 implementation: need check
  Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>> out_mat(
      out_blob->dptr<float>(), out.At(4), out.elem_cnt() / out.At(4));
  Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>> in_mat(
      in_blob->dptr<float>(), in.At(4), in.elem_cnt() / in.At(4));
  Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>
  out_diff_mat(out_diff_blob->dptr<float>(), out.At(4),
               out.elem_cnt() / out.At(4));
  Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>> in_diff_mat(
      in_diff_blob->mut_dptr<float>(), in.At(4), in.elem_cnt() / in.At(4));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(1)) {
      int64_t dstart = pd * strides.At(0) - padding_before.At(0);
      int64_t dend = std::min(dstart + pool_size.At(0), in.At(1));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(2)) {
        int64_t hstart = ph * strides.At(1) - padding_before.At(1);
        int64_t hend = std::min(hstart + pool_size.At(1), in.At(2));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(3)) {
          int64_t wstart = pw * strides.At(2) - padding_before.At(2);
          int64_t wend = std::min(wstart + pool_size.At(2), in.At(3));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int64_t pool_index =
              ((n * out.At(1) + pd) * out.At(2) + ph) * out.At(3) + pw;
          const float scale =
              1. / (hend - hstart) / (wend - wstart) / (dend - dstart);
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int64_t input_index =
                    ((n * in.At(1) + d) * in.At(2) + h) * in.At(3) + w;
                BackwardProcessGrad(pool_index, input_index, scale, out_mat,
                                    in_mat, out_diff_mat, in_diff_mat);
              }
            }
          }
        }
      }
    }
  }
}

#define INSTANTIATE_POOLING(type_cpp, type_proto) \
  template class Pooling<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
