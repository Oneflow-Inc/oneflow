#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void OneHotEncodeGpu(int64_t elem_cnt, const int64_t depth, const T on_value,
                                const T off_value, const K* indices, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t row = i / depth;
    const int64_t col = i - row * depth;
    const int64_t idx = indices[row];
    assert(idx >= 0 && idx < depth);
    out[i] = (idx == col) ? on_value : off_value;
  }
}

}  // namespace

template<typename T, typename K>
class GpuOneHotKernel final : public user_op::OpKernel {
 public:
  GpuOneHotKernel() = default;
  ~GpuOneHotKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_indices = indices->shape().elem_cnt();
    const int64_t depth = ctx->Attr<int64_t>("depth");
    const DataType dtype = ctx->Attr<DataType>("dtype");
    const T on_value = IsFloatingDataType(dtype)
                           ? static_cast<T>(ctx->Attr<double>("floating_on_value"))
                           : static_cast<T>(ctx->Attr<int64_t>("integer_on_value"));
    const T off_value = IsFloatingDataType(dtype)
                            ? static_cast<T>(ctx->Attr<double>("floating_off_value"))
                            : static_cast<T>(ctx->Attr<int64_t>("integer_off_value"));
    RUN_CUDA_KERNEL((OneHotEncodeGpu<T, K>), ctx->device_ctx(), num_indices * depth,
                    num_indices * depth, depth, on_value, off_value, indices->dptr<K>(),
                    out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ONE_HOT_KERNEL(dtype, itype)                                                \
  REGISTER_USER_KERNEL("one_hot").SetCreateFn<GpuOneHotKernel<dtype, itype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                                 \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0);  \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);          \
        return ctx.device_type() == DeviceType::kGPU                                             \
               && out_desc->data_type() == GetDataType<dtype>::value                             \
               && indices_desc->data_type() == GetDataType<itype>::value;                        \
      });

REGISTER_GPU_ONE_HOT_KERNEL(int32_t, int32_t)
REGISTER_GPU_ONE_HOT_KERNEL(int32_t, int64_t)
REGISTER_GPU_ONE_HOT_KERNEL(int64_t, int32_t)
REGISTER_GPU_ONE_HOT_KERNEL(int64_t, int64_t)
REGISTER_GPU_ONE_HOT_KERNEL(float, int32_t)
REGISTER_GPU_ONE_HOT_KERNEL(float, int64_t)
REGISTER_GPU_ONE_HOT_KERNEL(double, int32_t)
REGISTER_GPU_ONE_HOT_KERNEL(double, int64_t)

}  // namespace oneflow
