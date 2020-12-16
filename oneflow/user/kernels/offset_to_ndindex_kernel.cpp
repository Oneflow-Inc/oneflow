#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/kernels/offset_to_ndindex_util.h"

namespace oneflow{

namespace user_op{

template<DeviceType device_type, typename T>
class OffsetToNdIndexKernel final : public OpKernel {
 public:
  OffsetToNdIndexKernel() = default;
  ~OffsetToNdIndexKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* dims_tensor = ctx->Tensor4ArgNameAndIndex("dims", 0);
    T* dims = dims_tensor->dptr<T>();

    int ndim = dims_tensor->shape().elem_cnt();
    int32_t in_num = index->shape().elem_cnt(); 
    
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    
    OffsetToNdIndexFunctor<device_type, T>()(ctx->device_ctx(), in_num, ndim, index->dptr<T>(), dims, output);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_OFFSET_TO_NDINDEX_KERNEL(device, dtype)    \
  REGISTER_USER_KERNEL("offset_to_ndindex")                 \
      .SetCreateFn<OffsetToNdIndexKernel<device, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) & \
                        (user_op::HobDataType("dims", 0) == GetDataType<dtype>::value));

#define REGISTER_OFFSET_TO_NDINDEX_KERNELS_WITH_DEVICE(device) \
  REGISTER_OFFSET_TO_NDINDEX_KERNEL(device, int32_t)           \
  REGISTER_OFFSET_TO_NDINDEX_KERNEL(device, int64_t)           \

// Register CPU version
REGISTER_OFFSET_TO_NDINDEX_KERNELS_WITH_DEVICE(DeviceType::kCPU);


}  // namespace user_op
}  // namespace oneflow


