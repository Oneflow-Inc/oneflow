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

void Pooling3DCtx::BuildCudnnDescs(PoolingMode mode, DataType type) {
#ifdef WITH_CUDA
  std::vector<int> window{pool_size_d_, pool_size_h_, pool_size_w_};
  std::vector<int> padding{padding_d_, padding_h_, padding_w_};
  std::vector<int> stride{strides_d_, strides_h_, strides_w_};
  std::vector<int> full_stride{1, 1, strides_d_, strides_h_, strides_w_};
  std::vector<int> in_dim{in_n_, in_c_, in_d_, in_h_, in_w_};
  std::vector<int> out_dim{out_n_, out_c_, out_d_, out_h_, out_w_};

  pooling_desc_ = new CudnnPoolingNdDesc(mode, window, padding, stride);
  in_desc_ = new CudnnTensorNdDesc(type, in_dim, full_stride);
  out_desc_ = new CudnnTensorNdDesc(type, out_dim, full_stride);
  in_diff_desc_ = new CudnnTensorNdDesc(type, in_dim, full_stride);
  out_diff_desc_ = new CudnnTensorNdDesc(type, out_dim, full_stride);
#endif  // WITH_CUDA
}

}  // namespace oneflow
