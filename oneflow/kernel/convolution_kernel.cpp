#include "kernel/convolution_kernel.h"
#include "kernel/kernel_manager.h"
#include "job/resource.pb.h"
#include "register/blob.h"

namespace oneflow {

template<typename Dtype>
class ConvolutionKernel<DeviceType::kCPU, Dtype> final : public Kernel<Dtype> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) { TODO() };
  void Forward(
      std::function<Blob<Dtype>*(const std::string bn_in_op)>) {
    TODO()
  };
  void Backward(
      std::function<Blob<Dtype>*(const std::string bn_in_op)>) {
    TODO()
  };
};

REGISTER_CPU_KERNEL(OperatorConf::kConvolutionConf,
                          ConvolutionKernel);

}  // namespace oneflow
