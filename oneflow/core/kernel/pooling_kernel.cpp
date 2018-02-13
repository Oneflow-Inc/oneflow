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

void Pooling3DCtx::InitFromKernelConf(const Pooling3DKernelConf& kernel_conf) {
  kernel_conf_ = kernel_conf;
}

void Pooling3DCtx::BuildCudnnDescs(PoolingMode mode, DataType type) {
#ifdef WITH_CUDA
  std::vector<int> window = GetShapeInStdVec("pool_size");
  std::vector<int> padding = GetShapeInStdVec("padding_before");
  std::vector<int> stride = GetShapeInStdVec("strides");
  std::vector<int> full_stride{1, 1, stride.at(0), stride.at(1), stride.at(2)};
  std::vector<int> in_dim = GetShapeInStdVec("in");
  std::vector<int> out_dim = GetShapeInStdVec("out");

  pooling_desc_ = new CudnnPoolingNdDesc(mode, window, padding, stride);
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
class Pooling3DKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernelUtil);
  Pooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& kernel_ctx, const Blob* in_blob,
                      Blob* out_blob, const Pooling3DCtx& pooling_ctx) {
    TODO();
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
