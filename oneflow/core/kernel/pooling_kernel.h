#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

enum PoolingMode { kAveragePooling, kMaxPooling };

#ifdef WITH_CUDA
class CudnnPoolingNdDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingNdDesc);
  CudnnPoolingNdDesc() = delete;
  ~CudnnPoolingNdDesc();

  CudnnPoolingNdDesc(PoolingMode pooling_mode, const std::vector<int>& window,
                     const std::vector<int>& padding,
                     const std::vector<int>& stride);

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};
#endif

class Pooling3DCtx {
 public:
  Pooling3DCtx() = default;
  ~Pooling3DCtx();

  void InitFromKernelConf(const Pooling3DKernelConf& kernel_conf);
  void BuildCudnnDescs(PoolingMode mode, DataType type);

#ifdef WITH_CUDA
  CudnnTensorDesc* in_desc_ptr() const { return in_desc_; }
  CudnnTensorDesc* in_diff_desc_ptr() const { return in_diff_desc_; }
  CudnnTensorDesc* out_desc_ptr() const { return out_desc_; }
  CudnnTensorDesc* out_diff_desc_ptr() const { return out_diff_desc_; }
  CudnnPoolingNdDesc* pooling_desc_ptr() const { return pooling_desc_; }
#endif  // WITH_CUDA

 private:
  Pooling3DKernelConf kernel_conf_;
#ifdef WITH_CUDA
  CudnnTensorDesc* in_desc_;
  CudnnTensorDesc* in_diff_desc_;
  CudnnTensorDesc* out_desc_;
  CudnnTensorDesc* out_diff_desc_;
  CudnnPoolingNdDesc* pooling_desc_;
#endif  // WITH_CUDA
};

template<DeviceType device_type>
class PoolingKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  virtual PoolingMode GetPoolingMode() = 0;
  const Pooling3DCtx& pooling_3d_ctx() const { return pooling_3d_ctx_; }
  Pooling3DCtx* mut_pooling_3d_ctx() { return &pooling_3d_ctx_; }

  Pooling3DCtx pooling_3d_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
