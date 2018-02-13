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

  void Init(const Pooling3DKernelConf& kernel_conf, PoolingMode mode);
  void BuildCudnnDescs(DataType type);
  PoolingMode pooling_mode() const { return pooling_mode_; }
  const Pooling3DKernelConf& kernel_conf() const { return kernel_conf_; }

#ifdef WITH_CUDA
  CudnnTensorDesc* in_desc_ptr() const { return in_desc_; }
  CudnnTensorDesc* in_diff_desc_ptr() const { return in_diff_desc_; }
  CudnnTensorDesc* out_desc_ptr() const { return out_desc_; }
  CudnnTensorDesc* out_diff_desc_ptr() const { return out_diff_desc_; }
  CudnnPoolingNdDesc* pooling_desc_ptr() const { return pooling_desc_; }
#endif  // WITH_CUDA

 private:
  Pooling3DKernelConf kernel_conf_;
  PoolingMode pooling_mode_;
  std::vector<int> GetShapeInStdVec(const std::string& field_name) const;
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

template<DeviceType device_type, typename T>
class Pooling3DKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernelUtil);
  Pooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx&, const Blob*, Blob*,
                      const Pooling3DCtx&);

  static void Backward(const KernelCtx&, const Blob*, const Blob*, const Blob*,
                       Blob*, const Pooling3DCtx&);
};

template<typename T, typename PoolType>
void ForwardOnCPUWithOrderNCDHW(const Pooling3DCtx& pooling_ctx,
                                const Blob* in_blob, Blob* out_blob);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
