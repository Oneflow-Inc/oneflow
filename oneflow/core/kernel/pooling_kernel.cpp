#include "oneflow/core/kernel/pooling_kernel.h"

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
  delete in_diff_desc_;
  delete out_desc_;
  delete out_diff_desc_;
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
  std::vector<int> full_stride{1, 1, stride.at(0), stride.at(1), stride.at(2)};
  std::vector<int> in_dim = GetShapeInStdVec("in");
  std::vector<int> out_dim = GetShapeInStdVec("out");

  pooling_desc_ =
      new CudnnPoolingNdDesc(pooling_mode_, window, padding, stride);
  in_desc_ = new CudnnTensorDesc(type, in_dim, full_stride);
  out_desc_ = new CudnnTensorDesc(type, out_dim, full_stride);
  in_diff_desc_ = new CudnnTensorDesc(type, in_dim, full_stride);
  out_diff_desc_ = new CudnnTensorDesc(type, out_dim, full_stride);
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

  static void Process(const T& x_data, T& y_data) { y_data += x_data; }

  static void Finalize(const int size, T& y_data) { y_data /= size; }
};

template<typename T>
class MaxPoolForward {
 public:
  static T Initialize() { return std::numeric_limits<T>::min(); }

  static void Process(const T& x_data, T& y_data) {
    if (x_data > y_data) { y_data = x_data; }
  }

  static void Finalize(const int size, T& y_data) {}
};

template<typename T, typename PoolType>
void ForwardOnCPUWithOrderNCDHW(const Pooling3DCtx& ctx, const Blob* in_blob,
                                Blob* out_blob) {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  Shape pool_size(ctx.kernel_conf().pool_size());
  Shape strides(ctx.kernel_conf().strides());
  Shape padding_before(ctx.kernel_conf().padding_before());
  Shape padding_after(ctx.kernel_conf().padding_after());

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

template<typename T>
class Pooling3DKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernelUtil);
  Pooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& kernel_ctx, const Blob* in_blob,
                      Blob* out_blob, const Pooling3DCtx& pooling_ctx) {
    if (pooling_ctx.pooling_mode() == PoolingMode::kAveragePooling) {
      ForwardOnCPUWithOrderNCDHW<T, AveragePoolForward<T>>(pooling_ctx, in_blob,
                                                           out_blob);
    } else if (pooling_ctx.pooling_mode() == PoolingMode::kMaxPooling) {
      ForwardOnCPUWithOrderNCDHW<T, MaxPoolForward<T>>(pooling_ctx, in_blob,
                                                       out_blob);
    } else {
      UNEXPECTED_RUN();
    }
  }

  static void Backward(const KernelCtx& kernel_ctx, const Blob* out_diff_blob,
                       const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob, const Pooling3DCtx& pooling_ctx) {
    TODO();
  }
};

#define INSTANTIATE_POOLING_3D_KERNEL_UTIL(type_cpp, type_proto) \
  template class Pooling3DKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_3D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
