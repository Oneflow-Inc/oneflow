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
    Blob* in_diff_blob = BnInOp2Blob("in_diff");
    if (in_diff_blob == nullptr) { return; }
    Memset<device_type>(kernel_ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                        in_diff_blob->ByteSizeOfDataContentField());
    const Blob* out_diff_blob = BnInOp2Blob("out_diff");
    const Blob* in_blob = BnInOp2Blob("in");
    const Blob* out_blob = BnInOp2Blob("out");
    PoolingBackward(kernel_ctx, this->pooling_ctx(), out_diff_blob, out_blob, in_blob,
                    in_diff_blob);
  }
  virtual void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCtx& pooling_ctx,
                               const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                               Blob* in_diff_blob) const = 0;

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
                       const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob) const override;
  virtual void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                                T& in_diff) const = 0;
  virtual void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                                ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                                ConstEigenArrayMap<T>& out_diff_arr,
                                EigenArrayMap<T>& in_diff_arr) const = 0;
  void BackwardNCDHW(const PoolingCtx& pooling_ctx, const Blob* out_diff_blob, const Blob* out_blob,
                     const Blob* in_blob, Blob* in_diff_blob) const;
  void BackwardNDHWC(const PoolingCtx& pooling_ctx, const Blob* out_diff_blob, const Blob* out_blob,
                     const Blob* in_blob, Blob* in_diff_blob) const;
};

template<typename T>
class PoolingGradKernel<DeviceType::kGPU, T> : public PoolingGradKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingGradKernel);
  PoolingGradKernel() = default;
  virtual ~PoolingGradKernel() = default;

 protected:
  void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCtx& pooling_ctx,
                       const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_GRAD_KERNEL_H_
