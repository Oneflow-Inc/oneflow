#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void LeakyReluForwardGpu(const int n, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha; }
}

template<typename T>
__global__ void LeakyReluBackwardGpu(const int n, const float alpha, const T* x, const T* dy,
                                     T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = x[i] > 0 ? dy[i] : dy[i] * alpha; }
}

}  // namespace
template<typename T>
class LeakyReluKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LeakyReluKernel);
  LeakyReluKernel() = default;
  ~LeakyReluKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
  Blob* in_blob = BnInOp2Blob("in");
  const int32_t elem_cnt = in_blob->shape().elem_cnt();
  LeakyReluForwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(elem_cnt, this->op_conf().leaky_relu_conf().alpha(), in_blob->dptr<T>(),
                                                                                      BnInOp2Blob("out")->mut_dptr<T>());
}
};

template<typename T>
class LeakyReluGradKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LeakyReluGradKernel);
  LeakyReluGradKernel() = default;
  ~LeakyReluGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
  Blob* in_blob = BnInOp2Blob("x");
  const int32_t elem_cnt = in_blob->shape().elem_cnt();
  LeakyReluBackwardGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(elem_cnt, this->op_conf().leaky_relu_grad_conf().alpha(), in_blob->dptr<T>(),
                                                                                      BnInOp2Blob("dy")->dptr<T>(), BnInOp2Blob("dx")->mut_dptr<T>());
}
};


REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLeakyReluConf, DeviceType::kGPU, float,
                                      LeakyReluKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLeakyReluConf, DeviceType::kGPU, double,
                                      LeakyReluKernel<double>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLeakyReluGradConf, DeviceType::kGPU, float,
                                      LeakyReluGradKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLeakyReluGradConf, DeviceType::kGPU, double,
                                      LeakyReluGradKernel<double>)

}  // namespace oneflow

