#ifndef ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_
#define ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_

#include "kernel/kernel.h"
#include "job/resource.pb.h"

namespace oneflow {

template<DeviceType, typename Dtype>
class ConvolutionKernel final : public Kernel<Dtype> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override;
  void Forward(
      std::function<Blob<Dtype>*(const std::string bn_in_op)>) override;
  void Backward(
      std::function<Blob<Dtype>*(const std::string bn_in_op)>) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_
