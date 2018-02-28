#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "Eigen/Core"
#include "Eigen/Dense"

namespace oneflow {

#ifdef WITH_CUDA
class CudnnPoolingDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingDesc);
  CudnnPoolingDesc() = delete;
  ~CudnnPoolingDesc();

  CudnnPoolingDesc(cudnnPoolingMode_t pooling_mode,
                   const std::vector<int>& window,
                   const std::vector<int>& padding,
                   const std::vector<int>& stride);

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};
#endif  // WITH_CUDA

class Pooling3DCtx final {
 public:
  Pooling3DCtx(const Pooling3DKernelConf&);
#ifdef WITH_CUDA
  Pooling3DCtx(const Pooling3DKernelConf&, cudnnPoolingMode_t, DataType);
#endif  // WITH_CUDA
  ~Pooling3DCtx();

  const Pooling3DKernelConf& kernel_conf() const { return kernel_conf_; }

#ifdef WITH_CUDA
  CudnnTensorDesc* in_desc_ptr() const { return in_desc_; }
  CudnnTensorDesc* out_desc_ptr() const { return out_desc_; }
  CudnnPoolingDesc* pooling_desc_ptr() const { return pooling_desc_; }
#endif  // WITH_CUDA

 private:
  std::vector<int> GetShapeInStdVec(const std::string& field_name) const;
  Pooling3DKernelConf kernel_conf_;

#ifdef WITH_CUDA
  void BuildCudnnDescs(DataType type);
  cudnnPoolingMode_t pooling_mode_;
  CudnnTensorDesc* in_desc_;
  CudnnTensorDesc* out_desc_;
  CudnnPoolingDesc* pooling_desc_;
#endif  // WITH_CUDA
};

template<DeviceType device_type, typename T>
class PoolingKernelIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelIf);
  PoolingKernelIf() = default;
  virtual ~PoolingKernelIf() = default;

 protected:
#ifdef WITH_CUDA
  virtual cudnnPoolingMode_t GetCudnnPoolingMode() = 0;
#endif  // WITH_CUDA
  const Pooling3DCtx& pooling_3d_ctx() const { return *pooling_3d_ctx_; }
  void VirtualKernelInit(const ParallelContext*) override {
#ifdef WITH_CUDA
    pooling_3d_ctx_ =
        new Pooling3DCtx(GetPooling3DKernelConf(), this->GetCudnnPoolingMode(),
                         GetDataType<T>::val);
#else
    pooling_3d_ctx_ = new Pooling3DCtx(GetPooling3DKernelConf());
#endif  // WITH_CUDA
  }
  virtual const Pooling3DKernelConf& GetPooling3DKernelConf() const = 0;

  Pooling3DCtx* pooling_3d_ctx_;
};

template<DeviceType device_type, typename T>
class PoolingKernel : public PoolingKernelIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;
};

template<typename T>
class PoolingKernel<DeviceType::kCPU, T>
    : public PoolingKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  virtual void ForwardOnCPU(const Pooling3DCtx& pooling_ctx,
                            const Blob* in_blob, Blob* out_blob) const = 0;
  virtual T ForwardInitialize() const = 0;
  virtual void ForwardProcess(const T& lhs, T& rhs) const = 0;
  virtual void ForwardProcess(
      const int64_t in_col, const int64_t out_col,
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&
          in_mat,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
      const = 0;
  virtual void ForwardFinalize(const int64_t size, T& out) const = 0;
  virtual void ForwardFinalize(
      const int64_t size, const int64_t col,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
      const = 0;
  virtual void BackwardOnCPU(const Pooling3DCtx& pooling_ctax,
                             const Blob* out_diff_blob, const Blob* out_blob,
                             const Blob* in_blob, Blob* in_diff_blob) const = 0;
  virtual void BackwardProcessGrad(const T& in, const T& out, const T& out_diff,
                                   const float scale, T& in_diff) const = 0;
  virtual void BackwardProcessGrad(
      const int64_t out_col, const int64_t in_col, const float scale,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_diff_arr,
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_diff_arr) const = 0;
  void ForwardOnCPUWithOrderNCDHW(const Pooling3DCtx& pooling_ctx,
                                  const Blob* in_blob, Blob* out_blob) const;
  void BackwardOnCPUWithOrderNCDHW(const Pooling3DCtx& pooling_ctx,
                                   const Blob* out_diff_blob,
                                   const Blob* out_blob, const Blob* in_blob,
                                   Blob* in_diff_blob) const;
  void ForwardOnCPUWithOrderNDHWC(const Pooling3DCtx& pooling_ctx,
                                  const Blob* in_blob, Blob* out_blob) const;
  void BackwardOnCPUWithOrderNDHWC(const Pooling3DCtx& pooling_ctx,
                                   const Blob* out_diff_blob,
                                   const Blob* out_blob, const Blob* in_blob,
                                   Blob* in_diff_blob) const;
};

template<typename T>
class PoolingKernel<DeviceType::kGPU, T>
    : public PoolingKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
