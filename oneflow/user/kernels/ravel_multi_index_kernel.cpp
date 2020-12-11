#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/kernels/ravel_multi_index_util.h"

namespace oneflow {
 
namespace user_op {

template<DeviceType device_type, typename T>
class RavelMultiIndexKernel final : public OpKernel {
 public:
  RavelMultiIndexKernel() = default;
  ~RavelMultiIndexKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* dims_tensor = ctx->Tensor4ArgNameAndIndex("dims", 0);
    
    int ndim = dims_tensor->shape().elem_cnt();
    int32_t in_num = ctx->inputs().size() - 1; // ([3, 6, 2], [4, 5, 1]) -> in_num 还有个额外的输入 dim = 3, 所以要减1
  
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int32_t n = out->shape().elem_cnt();
    T* output = out->mut_dptr<T>();
    
    RavelMultiIndexFunctor<device_type, T>()(ctx->device_ctx(), ctx, n, in_num, ndim, dims_tensor, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RAVEL_MULTI_INDEX_KERNEL(device, dtype)    \
  REGISTER_USER_KERNEL("ravel_multi_index")                 \
      .SetCreateFn<RavelMultiIndexKernel<device, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) & \
                        (user_op::HobDataType("dims", 0) == GetDataType<dtype>::value));

#define REGISTER_RAVEL_MULTI_INDEX_KERNELS_WITH_DEVICE(device) \
  REGISTER_RAVEL_MULTI_INDEX_KERNEL(device, int32_t)           \
  REGISTER_RAVEL_MULTI_INDEX_KERNEL(device, int64_t)           \

// Register CPU version
REGISTER_RAVEL_MULTI_INDEX_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version

#ifdef WITH_CUDA
REGISTER_RAVEL_MULTI_INDEX_KERNELS_WITH_DEVICE(DeviceType::kGPU);
#endif

}  // namespace user_op
}  // namespace oneflow
