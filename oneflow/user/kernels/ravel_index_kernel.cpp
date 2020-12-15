#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/kernels/ravel_index_util.h"

namespace oneflow {
 
namespace user_op {

template<DeviceType device_type, typename T>
class RavelIndexKernel final : public OpKernel {
 public:
  RavelIndexKernel() = default;
  ~RavelIndexKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* dims_tensor = ctx->Tensor4ArgNameAndIndex("dims", 0);
    const Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
    
    int ndim = dims_tensor->shape().elem_cnt();
    int32_t in_num = index->shape().elem_cnt(); // ([3, 6, 2], [4, 5, 1]) -> in_num 
    
    std::cout<<"In num is: "<<in_num<<std::endl;

    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    
    RavelIndexFunctor<device_type, T>()(ctx->device_ctx(), in_num, ndim, index->dptr<T>(), dims_tensor->dptr<T>(), output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RAVEL_INDEX_KERNEL(device, dtype)    \
  REGISTER_USER_KERNEL("ravel_index")                 \
      .SetCreateFn<RavelIndexKernel<device, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) & \
                        (user_op::HobDataType("dims", 0) == GetDataType<dtype>::value));

#define REGISTER_RAVEL_INDEX_KERNELS_WITH_DEVICE(device) \
  REGISTER_RAVEL_INDEX_KERNEL(device, int32_t)           \
  REGISTER_RAVEL_INDEX_KERNEL(device, int64_t)           \

// Register CPU version
REGISTER_RAVEL_INDEX_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version

#ifdef WITH_CUDA
REGISTER_RAVEL_INDEX_KERNELS_WITH_DEVICE(DeviceType::kGPU);
#endif

}  // namespace user_op
}  // namespace oneflow
