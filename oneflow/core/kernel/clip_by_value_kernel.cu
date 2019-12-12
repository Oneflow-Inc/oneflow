#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int64_t elem_cnt, const T* in_ptr, const T min_val,
                           const T max_val, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out_ptr[i] = min(max(in_ptr[i], min_val), max_val); }
}

}  // namespace

template<typename T>
class ClipByValueGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByValueGpuKernel);
  ClipByValueGpuKernel() = default;
  ~ClipByValueGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");

    const auto& shape = in_blob->shape();
    CHECK(out_blob->shape() == shape);
    const ClipByValueOpConf& conf = this->op_conf().clip_by_value_conf();
    const T min_val = conf.has_min_val() ? static_cast<T>(conf.min_val()) : GetMinVal<T>();
    const T max_val = conf.has_max_val() ? static_cast<T>(conf.max_val()) : GetMaxVal<T>();
    GpuForward<<<BlocksNum4ThreadsNum(shape.elem_cnt()), kCudaThreadsNumPerBlock, 0,
                 ctx.device_ctx->cuda_stream()>>>(shape.elem_cnt(), in_blob->dptr<T>(), min_val,
                                                  max_val, out_blob->mut_dptr<T>());
  }
};

#define MAKE_ENTRY(type_cpp, type_proto)                                              \
  NEW_REGISTER_KERNEL(OperatorConf::kClipByValueConf, ClipByValueGpuKernel<type_cpp>) \
      .SetIsMatchedPred([](const KernelConf& conf) {                                  \
        return (DeviceType::kGPU == conf.op_attribute().op_conf().device_type())      \
               && (type_proto == conf.data_type());                                   \
      });

OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
