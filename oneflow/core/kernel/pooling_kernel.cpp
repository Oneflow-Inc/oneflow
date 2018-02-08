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
  std::vector<int> window{kernel_conf_.pool_size_d(),
                          kernel_conf_.pool_size_h(),
                          kernel_conf_.pool_size_w()};
  std::vector<int> padding{kernel_conf_.padding_d(), kernel_conf_.padding_h(),
                           kernel_conf_.padding_w()};
  std::vector<int> stride{kernel_conf_.strides_d(), kernel_conf_.strides_h(),
                          kernel_conf_.strides_w()};
  std::vector<int> full_stride{1, 1, kernel_conf_.strides_d(),
                               kernel_conf_.strides_h(),
                               kernel_conf_.strides_w()};
  std::vector<int> in_dim{kernel_conf_.in_shape(0), kernel_conf_.in_shape(1),
                          kernel_conf_.in_shape(2), kernel_conf_.in_shape(3),
                          kernel_conf_.in_shape(4)};
  std::vector<int> out_dim{kernel_conf_.out_shape(0), kernel_conf_.out_shape(1),
                           kernel_conf_.out_shape(2), kernel_conf_.out_shape(3),
                           kernel_conf_.out_shape(4)};

  pooling_desc_ = new CudnnPoolingNdDesc(mode, window, padding, stride);
  in_desc_ = new CudnnTensorNdDesc(type, in_dim, full_stride);
  out_desc_ = new CudnnTensorNdDesc(type, out_dim, full_stride);
  in_diff_desc_ = new CudnnTensorNdDesc(type, in_dim, full_stride);
  out_diff_desc_ = new CudnnTensorNdDesc(type, out_dim, full_stride);
#endif  // WITH_CUDA
}

}  // namespace oneflow
