#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

namespace oneflow {

#ifdef WITH_CUDA
class CudnnPoolingNdDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingNdDesc);
  CudnnPoolingNdDesc() = delete;
  ~CudnnPoolingNdDesc();

  CudnnPoolingNdDesc(cudnnPoolingMode_t pooling_mode,
                     const std::vector<int>& window,
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

  void set_kernel_conf(const Pooling3DKernelConf& kernel_conf);
#ifdef WITH_CUDA
  void set_cudnn_pooling_mode(cudnnPoolingMode_t pooling_mode);
  void BuildCudnnDescs(DataType type);
#endif  // WITH_CUDA
  const Pooling3DKernelConf& kernel_conf() const { return kernel_conf_; }

#ifdef WITH_CUDA
  CudnnTensorDesc* in_desc_ptr() const { return in_desc_; }
  CudnnTensorDesc* out_desc_ptr() const { return out_desc_; }
  CudnnPoolingNdDesc* pooling_desc_ptr() const { return pooling_desc_; }
#endif  // WITH_CUDA

 private:
  Pooling3DKernelConf kernel_conf_;
  std::vector<int> GetShapeInStdVec(const std::string& field_name) const;
#ifdef WITH_CUDA
  cudnnPoolingMode_t pooling_mode_;
  CudnnTensorDesc* in_desc_;
  CudnnTensorDesc* out_desc_;
  CudnnPoolingNdDesc* pooling_desc_;
#endif  // WITH_CUDA
};

template<DeviceType device_type, typename T>
class PoolingIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingIf);
  PoolingIf() = default;
  virtual ~PoolingIf() = default;

 protected:
#ifdef WITH_CUDA
  virtual cudnnPoolingMode_t GetCudnnPoolingMode() = 0;
#endif  // WITH_CUDA
  const Pooling3DCtx& pooling_3d_ctx() const { return pooling_3d_ctx_; }
  Pooling3DCtx* mut_pooling_3d_ctx() { return &pooling_3d_ctx_; }
  void VirtualKernelInit(const ParallelContext*) override {
    Pooling3DCtx* pooling_3d_ctx = this->mut_pooling_3d_ctx();
    pooling_3d_ctx->set_kernel_conf(GetPooling3DKernelConf());
#ifdef WITH_CUDA
    pooling_3d_ctx->set_cudnn_pooling_mode(this->GetCudnnPoolingMode());
    pooling_3d_ctx->BuildCudnnDescs(GetDataType<T>::val);
#endif  // WITH_CUDA
  }
  virtual const Pooling3DKernelConf& GetPooling3DKernelConf() const = 0;

  Pooling3DCtx pooling_3d_ctx_;
};

template<DeviceType device_type, typename T>
class Pooling : public PoolingIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling);
  Pooling() = default;
  virtual ~Pooling() = default;
};

template<typename T>
class Pooling<DeviceType::kCPU, T> : public PoolingIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling);
  Pooling() = default;
  virtual ~Pooling() = default;

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
class Pooling<DeviceType::kGPU, T> : public PoolingIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling);
  Pooling() = default;
  virtual ~Pooling() = default;

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
