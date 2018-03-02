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
  Pooling3DCtx(const Pooling3DKernelConf&
#ifdef WITH_CUDA
               ,
               cudnnPoolingMode_t, DataType
#endif  // WITH_CUDA
  );
  ~Pooling3DCtx();

  const Pooling3DKernelConf& kernel_conf() const { return kernel_conf_; }

#ifdef WITH_CUDA
  const cudnnTensorDescriptor_t& cudnn_in_tensor_desc() const;
  const cudnnTensorDescriptor_t& cudnn_out_tensor_desc() const;
  const cudnnPoolingDescriptor_t& cudnn_pooling_desc() const;
#endif  // WITH_CUDA

 private:
  std::vector<int> GetStdVecFromShapeInKernelConf(
      const std::string& field_name) const;
  Pooling3DKernelConf kernel_conf_;

#ifdef WITH_CUDA
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
    pooling_3d_ctx_ =
        new Pooling3DCtx(GetPooling3DKernelConf()
#ifdef WITH_CUDA
                             ,
                         this->GetCudnnPoolingMode(), GetDataType<T>::val
#endif  // WITH_CUDA
        );
  }
  virtual const Pooling3DKernelConf& GetPooling3DKernelConf() const = 0;
  void ForwardDataContent(
      const KernelCtx& kernel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    Forward(kernel_ctx, this->pooling_3d_ctx(), in_blob, out_blob);
  }
  void BackwardDataContent(
      const KernelCtx& kernel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* in_diff_blob = BnInOp2Blob("in_diff");
    if (in_diff_blob == nullptr) { return; }
    Memset<device_type>(kernel_ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                        in_diff_blob->ByteSizeOfDataContentField());
    const Blob* out_diff_blob = BnInOp2Blob("out_diff");
    const Blob* in_blob = BnInOp2Blob("in");
    const Blob* out_blob = BnInOp2Blob("out");
    Backward(kernel_ctx, this->pooling_3d_ctx(), out_diff_blob, out_blob,
             in_blob, in_diff_blob);
  }
  virtual void Forward(const KernelCtx& kernel_ctx,
                       const Pooling3DCtx& pooling_ctx, const Blob* in_blob,
                       Blob* out_blob) const = 0;
  virtual void Backward(const KernelCtx& kernel_ctx,
                        const Pooling3DCtx& pooling_ctx,
                        const Blob* out_diff_blob, const Blob* out_blob,
                        const Blob* in_blob, Blob* in_diff_blob) const = 0;

  Pooling3DCtx* pooling_3d_ctx_;
};

template<DeviceType device_type, typename T>
class PoolingKernel;

template<typename T>
class PoolingKernel<DeviceType::kCPU, T>
    : public PoolingKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  virtual T ForwardInitialize() const = 0;
  virtual void NCDHWProcess(const T& lhs, T& rhs) const = 0;
  virtual void NDHWCProcess(
      const int64_t in_col, const int64_t out_col,
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&
          in_mat,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
      const = 0;
  virtual void NCDHWFinalize(const int64_t size, T& out) const = 0;
  virtual void NDHWCFinalize(
      const int64_t size, const int64_t col,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
      const = 0;
  virtual void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff,
                                const float scale, T& in_diff) const = 0;
  virtual void NDHWCProcessGrad(
      const int64_t out_col, const int64_t in_col, const float scale,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_diff_arr,
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_diff_arr) const = 0;
  void ForwardNCDHW(const Pooling3DCtx& pooling_ctx, const Blob* in_blob,
                    Blob* out_blob) const;
  void BackwardNCDHW(const Pooling3DCtx& pooling_ctx, const Blob* out_diff_blob,
                     const Blob* out_blob, const Blob* in_blob,
                     Blob* in_diff_blob) const;
  void ForwardNDHWC(const Pooling3DCtx& pooling_ctx, const Blob* in_blob,
                    Blob* out_blob) const;
  void BackwardNDHWC(const Pooling3DCtx& pooling_ctx, const Blob* out_diff_blob,
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
  void Forward(const KernelCtx& kernel_ctx, const Pooling3DCtx& pooling_ctx,
               const Blob* in_blob, Blob* out_blob) const override;
  void Backward(const KernelCtx& kernel_ctx, const Pooling3DCtx& pooling_ctx,
                const Blob* out_diff_blob, const Blob* out_blob,
                const Blob* in_blob, Blob* in_diff_blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
