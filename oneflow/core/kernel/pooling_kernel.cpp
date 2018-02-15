#include "oneflow/core/kernel/pooling_kernel.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

namespace oneflow {

#ifdef WITH_CUDA
CudnnPoolingNdDesc::~CudnnPoolingNdDesc() {
  CudaCheck(cudnnDestroyPoolingDescriptor(val_));
}

CudnnPoolingNdDesc::CudnnPoolingNdDesc(PoolingMode pooling_mode,
                                       const std::vector<int>& window,
                                       const std::vector<int>& padding,
                                       const std::vector<int>& stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(
      val_,
      (pooling_mode == PoolingMode::kAveragePooling
           ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
           : CUDNN_POOLING_MAX),
      CUDNN_NOT_PROPAGATE_NAN, window.size(), window.data(), padding.data(),
      stride.data()));
}
#endif

Pooling3DCtx::~Pooling3DCtx() {
#ifdef WITH_CUDA
  delete in_desc_;
  delete out_desc_;
  delete pooling_desc_;
#endif  // WITH_CUDA
}

void Pooling3DCtx::Init(const Pooling3DKernelConf& kernel_conf,
                        PoolingMode pooling_mode) {
  kernel_conf_ = kernel_conf;
  pooling_mode_ = pooling_mode;
}

void Pooling3DCtx::BuildCudnnDescs(DataType type) {
#ifdef WITH_CUDA
  std::vector<int> window = GetShapeInStdVec("pool_size");
  std::vector<int> padding = GetShapeInStdVec("padding_before");
  std::vector<int> stride = GetShapeInStdVec("strides");
  std::vector<int> in_dim = GetShapeInStdVec("in");
  std::vector<int> in_stride{in_dim[1] * in_dim[2] * in_dim[3] * in_dim[4],
                             in_dim[2] * in_dim[3] * in_dim[4],
                             in_dim[3] * in_dim[4], in_dim[4], 1};
  std::vector<int> out_dim = GetShapeInStdVec("out");
  std::vector<int> out_stride = {
      out_dim[1] * out_dim[2] * out_dim[3] * out_dim[4],
      out_dim[2] * out_dim[3] * out_dim[4], out_dim[3] * out_dim[4], out_dim[4],
      1};

  pooling_desc_ =
      new CudnnPoolingNdDesc(pooling_mode_, window, padding, stride);
  in_desc_ = new CudnnTensorDesc(type, in_dim, in_stride);
  out_desc_ = new CudnnTensorDesc(type, out_dim, out_stride);
#endif  // WITH_CUDA
}

std::vector<int> Pooling3DCtx::GetShapeInStdVec(
    const std::string& field_name) const {
  PbRf<int64_t> shape = GetPbRfFromPbMessage<int64_t>(
      GetMessageFromPbMessage(kernel_conf_, field_name), "dim");
  std::vector<int> ret;
  FOR_RANGE(size_t, i, 0, shape.size()) { ret.push_back(shape.Get(i)); }
  return ret;
}

template<typename T>
class AveragePoolForward {
 public:
  static T Initialize() { return static_cast<T>(0); }

  static void Process(const T& lhs, T& rhs) { rhs += lhs; }

  static void Process(
      const int64_t in_col, const int64_t out_col,
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&
          in_mat,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat) {
    out_mat.col(out_col) += in_mat.col(in_col);
  }

  static void Finalize(const int64_t size, T& out) { out /= size; }

  static void Finalize(
      const int64_t size, const int64_t col,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat) {
    out_mat.col(col) /= size;
  }
};

template<typename T>
class MaxPoolForward {
 public:
  static T Initialize() { return std::numeric_limits<T>::min(); }

  static void Process(const T& lhs, T& rhs) {
    if (lhs > rhs) { rhs = lhs; }
  }

  static void Process(
      const int64_t in_col, const int64_t out_col,
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&
          in_mat,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat) {
    out_mat.col(out_col) = out_mat.col(out_col).cwiseMax(in_mat.col(in_col));
  }

  static void Finalize(const int64_t size, T& out) {}

  static void Finalize(
      const int64_t size, const int64_t col,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat) {}
};

template<typename T, typename PoolType>
void ForwardOnCPUWithOrderNCDHW(const Pooling3DCtx& ctx, const Blob* in_blob,
                                Blob* out_blob) {
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
            T res = PoolType::Initialize();
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t input_index =
                      d * in.Count(3) + h * in.At(4) + w;
                  PoolType::Process(input[input_index], res);
                }
              }
            }
            PoolType::Finalize(
                (dend - dstart) * (hend - hstart) * (wend - wstart), res);
            output[pool_index] = res;
          }
        }
      }
      input += in.Count(2);
      output += out.Count(2);
    }
  }
}

template<typename T, typename PoolType>
void ForwardOnCPUWithOrderNDHWC(const Pooling3DCtx& ctx, const Blob* in_blob,
                                Blob* out_blob) {
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
          out_mat.col(out_col).setConstant(PoolType::Initialize());
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int in_col =
                    ((n * in.At(1) + d) * in.At(2) + h) * in.At(3) + w;
                PoolType::Process(in_col, out_col, in_mat, out_mat);
              }
            }
          }
          PoolType::Finalize(
              (hend - hstart) * (wend - wstart) * (dend - dstart), out_col,
              out_mat);
        }
      }
    }
  }
}

template<typename T>
class AveragePoolBackward {
 public:
  static void ProcessGrad(const T&, const T&, const T& out_diff,
                          const float scale, T& in_diff) {
    in_diff += (scale * out_diff);
  }

  static void ProcessGrad(
      const int64_t out_col, const int64_t in_col, const float scale,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_diff_arr,
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_diff_arr) {
    in_diff_arr.col(in_col) += scale * out_diff_arr.col(out_col);
  }
};

template<typename T>
class MaxPoolBackward {
 public:
  static void ProcessGrad(const T& in, const T& out, const T& out_diff,
                          const float, T& in_diff) {
    if (in == out) { in_diff += out_diff; }
  }

  static void ProcessGrad(
      const int64_t out_col, const int64_t in_col, const float scale,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_diff_arr,
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_diff_arr) {
    in_diff_arr.col(in_col) += out_diff_arr.col(out_col)
                               * (in_arr.col(in_col)
                                      .cwiseEqual(out_arr.col(out_col))
                                      .template cast<float>());
  }
};

template<typename T, typename PoolType>
void BackwardOnCPUWithOrderNCDHW(const Pooling3DCtx& ctx,
                                 const Blob* out_diff_blob,
                                 const Blob* out_blob, const Blob* in_blob,
                                 Blob* in_diff_blob) {
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
                  PoolType::ProcessGrad(input[index], output[pool_index],
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

template<typename T, typename PoolType>
void BackwardOnCPUWithOrderNDHWC(const Pooling3DCtx& ctx,
                                 const Blob* out_diff_blob,
                                 const Blob* out_blob, const Blob* in_blob,
                                 Blob* in_diff_blob) {
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
                PoolType::ProcessGrad(pool_index, input_index, scale, out_mat,
                                      in_mat, out_diff_mat, in_diff_mat);
              }
            }
          }
        }
      }
    }
  }
}

template<typename T>
class Pooling3DKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernelUtil);
  Pooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& kernel_ctx, const Blob* in_blob,
                      Blob* out_blob, const Pooling3DCtx& pooling_ctx) {
    const std::string& data_format = pooling_ctx.kernel_conf().data_format();
    if (pooling_ctx.pooling_mode() == PoolingMode::kAveragePooling) {
      if (data_format == "channels_first") {
        ForwardOnCPUWithOrderNCDHW<T, AveragePoolForward<T>>(pooling_ctx,
                                                             in_blob, out_blob);
      } else if (data_format == "channels_last") {
        ForwardOnCPUWithOrderNDHWC<T, AveragePoolForward<T>>(pooling_ctx,
                                                             in_blob, out_blob);
      } else {
        UNEXPECTED_RUN();
      }
    } else if (pooling_ctx.pooling_mode() == PoolingMode::kMaxPooling) {
      if (data_format == "channels_first") {
        ForwardOnCPUWithOrderNCDHW<T, MaxPoolForward<T>>(pooling_ctx, in_blob,
                                                         out_blob);
      } else if (data_format == "channels_last") {
        ForwardOnCPUWithOrderNDHWC<T, MaxPoolForward<T>>(pooling_ctx, in_blob,
                                                         out_blob);
      } else {
        UNEXPECTED_RUN();
      }
    } else {
      UNEXPECTED_RUN();
    }
  }

  static void Backward(const KernelCtx& kernel_ctx, const Blob* out_diff_blob,
                       const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob, const Pooling3DCtx& pooling_ctx) {
    const std::string& data_format = pooling_ctx.kernel_conf().data_format();
    if (pooling_ctx.pooling_mode() == PoolingMode::kAveragePooling) {
      if (data_format == "channels_first") {
        BackwardOnCPUWithOrderNCDHW<T, AveragePoolBackward<T>>(
            pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
      } else if (data_format == "channels_last") {
        BackwardOnCPUWithOrderNDHWC<T, AveragePoolBackward<T>>(
            pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
      } else {
        UNEXPECTED_RUN();
      }
    } else if (pooling_ctx.pooling_mode() == PoolingMode::kMaxPooling) {
      if (data_format == "channels_first") {
        BackwardOnCPUWithOrderNCDHW<T, MaxPoolBackward<T>>(
            pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
      } else if (data_format == "channels_last") {
        BackwardOnCPUWithOrderNDHWC<T, MaxPoolBackward<T>>(
            pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
      } else {
        UNEXPECTED_RUN();
      }
    } else {
      UNEXPECTED_RUN();
    }
  }
};

#define INSTANTIATE_POOLING_3D_KERNEL_UTIL(type_cpp, type_proto) \
  template class Pooling3DKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_3D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
