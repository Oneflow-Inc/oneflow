#ifndef ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PoolingGradKernelIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGradKernelIf);
  PoolingGradKernelIf() = default;
  virtual ~PoolingGradKernelIf() = default;

 protected:
#ifdef WITH_CUDA
  virtual cudnnPoolingMode_t GetCudnnPoolingMode() = 0;
#endif  // WITH_CUDA
  const PoolingCtx& pooling_ctx() const { return *pooling_ctx_; }
  void VirtualKernelInit(const ParallelContext*) override {
    pooling_ctx_.reset(new PoolingCtx(GetPoolingKernelConf()
#ifdef WITH_CUDA
                                          ,
                                      this->GetCudnnPoolingMode(), GetDataType<T>::value
#endif  // WITH_CUDA
                                      ));
  }
  virtual const PoolingKernelConf& GetPoolingKernelConf() const = 0;
  void ForwardDataContent(const KernelCtx& kernel_ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* dx_blob = BnInOp2Blob("dx");
    if (dx_blob == nullptr) { return; }
    Memset<device_type>(kernel_ctx.device_ctx, dx_blob->mut_dptr(), 0,
                        dx_blob->ByteSizeOfDataContentField());
    const Blob* dy_blob = BnInOp2Blob("dy");
    const Blob* x_blob = BnInOp2Blob("x");
    const Blob* y_blob = BnInOp2Blob("y");
    PoolingBackward(kernel_ctx, this->pooling_ctx(), dy_blob, y_blob, x_blob, dx_blob);
  }
  virtual void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCtx& pooling_ctx,
                               const Blob* dy_blob, const Blob* y_blob, const Blob* x_blob,
                               Blob* dx_blob) const = 0;

  std::unique_ptr<PoolingCtx> pooling_ctx_;
};

template<DeviceType device_type, typename T>
class PoolingGradKernel;

template<typename T>
class PoolingGradKernel<DeviceType::kCPU, T> : public PoolingGradKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGradKernel);
  PoolingGradKernel() = default;
  virtual ~PoolingGradKernel() = default;

 protected:
  void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCtx& pooling_ctx,
                       const Blob* dy_blob, const Blob* y_blob, const Blob* x_blob,
                       Blob* dx_blob) const override;
  virtual void NCDHWProcessGrad(const T& x, const T& y, const T& dy, const int64_t size,
                                T& dx) const = 0;
  virtual void NDHWCProcessGrad(const int64_t y_col, const int64_t x_col, const int64_t size,
                                ConstEigenArrayMap<T>& y_arr, ConstEigenArrayMap<T>& x_arr,
                                ConstEigenArrayMap<T>& dy_arr, EigenArrayMap<T>& dx_arr) const = 0;
  void BackwardNCDHW(const PoolingCtx& pooling_ctx, const Blob* dy_blob, const Blob* y_blob,
                     const Blob* x_blob, Blob* dx_blob) const;
  void BackwardNDHWC(const PoolingCtx& pooling_ctx, const Blob* dy_blob, const Blob* y_blob,
                     const Blob* x_blob, Blob* dx_blob) const;
};

template<typename T>
class PoolingGradKernel<DeviceType::kGPU, T> : public PoolingGradKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGradKernel);
  PoolingGradKernel() = default;
  virtual ~PoolingGradKernel() = default;

 protected:
  void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCtx& pooling_ctx,
                       const Blob* dy_blob, const Blob* y_blob, const Blob* x_blob,
                       Blob* dx_blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_
