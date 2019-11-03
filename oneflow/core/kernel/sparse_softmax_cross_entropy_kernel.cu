#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void SparseSoftmaxCrossEntropyGradBackwardSub(const int64_t n, const int64_t w,
                                                         const int64_t lower_bound, const T* dy,
                                                         const K* label, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int64_t idx = label[i] - lower_bound;
    if (idx >= 0 && idx < w) { in_diff[i * w + idx] = dy[i] * (in_diff[i * w + idx] - 1); }
  }
}

}  // namespace

template<typename T, typename K>
class SparseSoftmaxCrossEntropyGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyGpuKernel);
  SparseSoftmaxCrossEntropyGpuKernel() = default;
  ~SparseSoftmaxCrossEntropyGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* prediction_blob = BnInOp2Blob("prediction");
    const Blob* label_blob = BnInOp2Blob("label");
    Blob* tmp_blob = BnInOp2Blob("fw_softmax_num");
    Blob* buf_blob = BnInOp2Blob("fw_buf");
    Blob* prob_blob = BnInOp2Blob("prob");
    Blob* out_blob = BnInOp2Blob("out");
    const int64_t n = prediction_blob->shape().At(0);
    const int64_t w = prediction_blob->shape().Count(1);
    Memset<DeviceType::kGPU>(ctx.device_ctx, out_blob->mut_dptr(), 0,
                             out_blob->ByteSizeOfDataContentField());
    SoftmaxComputeProb<DeviceType::kGPU, T>(
        ctx.device_ctx, n, w, prediction_blob->dptr<T>(), tmp_blob->mut_dptr<T>(),
        prob_blob->mut_dptr<T>(), buf_blob->mut_dptr(), buf_blob->ByteSizeOfDataContentField());
    SparseCrossEntropyKernelUtil<DeviceType::kGPU, T, K>::ComputeEntropy(
        ctx.device_ctx, n, w, prob_blob->dptr<T>(), label_blob->dptr<K>(), out_blob->mut_dptr<T>());
  }
};

template<typename T, typename K>
class SparseSoftmaxCrossEntropyGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyGradGpuKernel);
  SparseSoftmaxCrossEntropyGradGpuKernel() = default;
  ~SparseSoftmaxCrossEntropyGradGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy_blob = BnInOp2Blob("dy");
    const Blob* label_blob = BnInOp2Blob("label");
    const Blob* prob_blob = BnInOp2Blob("prob");
    Blob* dx_blob = BnInOp2Blob("dx");
    int64_t lower_bound = 0;
    if (this->kernel_conf().has_sparse_softmax_cross_entropy_grad_conf()) {
      lower_bound = this->kernel_conf().sparse_softmax_cross_entropy_grad_conf().lower_bound();
    }
    const int64_t n = dx_blob->shape().At(0);
    const int64_t w = dx_blob->shape().Count(1);
    dx_blob->CopyDataContentFrom(ctx.device_ctx, prob_blob);
    SparseSoftmaxCrossEntropyGradBackwardSub<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                                               ctx.device_ctx->cuda_stream()>>>(
        n, w, lower_bound, dy_blob->dptr<T>(), label_blob->dptr<K>(), dx_blob->mut_dptr<T>());
  }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(dtype, ltype)      \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyConf,                  \
                      SparseSoftmaxCrossEntropyGpuKernel<dtype, ltype>)              \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_conf().label_type()));      \
      });                                                                            \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyMs1Conf,               \
                      SparseSoftmaxCrossEntropyGpuKernel<dtype, ltype>)              \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_conf().label_type()));      \
      });                                                                            \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyGradConf,              \
                      SparseSoftmaxCrossEntropyGradGpuKernel<dtype, ltype>)          \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_grad_conf().label_type())); \
      });                                                                            \
  NEW_REGISTER_KERNEL(OperatorConf::kSparseSoftmaxCrossEntropyMs1GradConf,           \
                      SparseSoftmaxCrossEntropyGradGpuKernel<dtype, ltype>)          \
      .SetIsMatchedPred([](const KernelConf& conf) {                                 \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)    \
                && (GetDataType<dtype>::value == conf.data_type())                   \
                && (GetDataType<ltype>::value                                        \
                    == conf.sparse_softmax_cross_entropy_grad_conf().label_type())); \
      });

REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(float, int64_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(double, int64_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(float, int32_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(double, int32_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(float, int8_t);
REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_AND_GRAD_GPU_KERNEL(double, int8_t);

}  // namespace oneflow