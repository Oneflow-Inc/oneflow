#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int64_t num_boxes, const T* in_ptr, const int32_t canonical_level,
                           const int32_t canonical_scale, const int32_t min_level,
                           const int32_t max_level, const float epsilon, int32_t* out_ptr) {
  const T TO_REMOVE = 1.0;
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    const T scale = sqrt((in_ptr[i * 4 + 2] - in_ptr[i * 4] + TO_REMOVE)
                         * (in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] + TO_REMOVE));
    const int32_t target_level = floor(canonical_level + log2(scale / canonical_scale + epsilon));
    out_ptr[i] = min(max(target_level, min_level), max_level) - min_level;
  }
}

}  // namespace

template<typename T>
class LevelMapGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LevelMapGPUKernel);
  LevelMapGPUKernel() = default;
  ~LevelMapGPUKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    int32_t* out = BnInOp2Blob("out")->mut_dptr<int32_t>();

    CHECK_EQ(in_blob->shape().dim_vec().back(), 4);
    int64_t num_boxes =
        std::accumulate(in_blob->shape().dim_vec().begin(), in_blob->shape().dim_vec().end() - 1, 1,
                        std::multiplies<int64_t>());
    const T* in_ptr = in_blob->dptr<T>();
    const auto conf = this->op_conf().level_map_conf();
    const int32_t canonical_level = conf.canonical_level();
    const float canonical_scale = conf.canonical_scale();
    const int32_t min_level = conf.min_level();
    const int32_t max_level = conf.max_level();
    const float epsilon = conf.epsilon();
    GpuForward<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
        num_boxes, in_ptr, canonical_level, canonical_scale, min_level, max_level, epsilon,
        out);
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLevelMapConf, DeviceType::kGPU, float,
                                      LevelMapGPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLevelMapConf, DeviceType::kGPU, double,
                                      LevelMapGPUKernel<double>)

}  // namespace oneflow
