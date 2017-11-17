#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PoolingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  ~PoolingKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class PoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx&, const Blob*, Blob*, Blob*,
                             const PoolingOpConf&);

  static void PoolingBackward(const KernelCtx&, const Blob*, const Blob*, Blob*,
                              const PoolingOpConf&);
};

template<DeviceType device_type, typename T>
class CudnnPoolingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingKernel);
  CudnnPoolingKernel();
  ~CudnnPoolingKernel();

  void InitFromOpProto(const OperatorProto& op_proto) override;
  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
class CudnnPoolingKernel<DeviceType::kCPU, T> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingKernel);
  CudnnPoolingKernel() = default;
  ~CudnnPoolingKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override {
    UNEXPECTED_RUN();
  }
  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }
};

template<typename T>
class CudnnPoolingKernel<DeviceType::kGPU, T> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingKernel);
  CudnnPoolingKernel();
  ~CudnnPoolingKernel();

  void InitFromOpProto(const OperatorProto& op_proto) override;
  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnPoolingMode_t pooling_mode_;
  cudnnPoolingDescriptor_t pooling_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
