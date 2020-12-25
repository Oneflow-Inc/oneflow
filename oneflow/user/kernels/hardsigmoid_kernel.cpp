#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow{

namespace user_op{

template<DeviceType device_type, typename T>
class CpuHardsigmoidKernel final : public OpKernel{
public: 
    CpuHardsigmoidKernel() = default;
    ~CpuHardsigmoidKernel() = default;

private: 
    void Compute(KernelComputeContext* ctx) const override{
        const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
        Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
        const T* in_ptr = in_tensor->dptr<T>();
        T* out_ptr = out_tensor->mut_dptr<T>();

        const int32_t elem_cnt = in_tensor->shape().elem_cnt();
        FOR_RANGE(int32_t, i, 0, elem_cnt){
            if(in_ptr[i] <= static_cast<T>(-3))
                out_ptr[i] = static_cast<T>(0);
            else if(in_ptr[i] >= static_cast<T>(3))
                out_ptr[i] = static_cast<T>(1);
            else 
                out_ptr[i] = (in_ptr[i] / static_cast<T>(6)) + (static_cast<T>(1) / static_cast<T>(2));
        }
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override {return false;}
};

#define REGISTER_CPU_HARDSIGMOID_KERNEL(device, dtype)\
    REGISTER_USER_KERNEL("hardsigmoid")                  \
        .SetCreateFn<CpuHardsigmoidKernel<device, dtype>>() \
        .SetIsMatchedHob((HobDeviceTag()==device) & (HobDataType("out", 0) == GetDataType<dtype>::value)) \
        .SetInplaceProposalFn(                                                              \
          [](const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));              \
            return Maybe<void>::Ok();                                                     \
          });

REGISTER_CPU_HARDSIGMOID_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_HARDSIGMOID_KERNEL(DeviceType::kCPU, double)

template<DeviceType device_type, typename T>
class CpuHardsigmoidGradKernel final: public OpKernel {
public:
    CpuHardsigmoidGradKernel() = default;
    ~CpuHardsigmoidGradKernel() = default;
    
private: 
    void Compute(KernelComputeContext *ctx) const override {
        const Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
        const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
        Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
        const T* y_ptr = y_tensor->dptr<T>();
        const T* dy_ptr = dy_tensor->dptr<T>();
        T* dx_ptr = dx_tensor->mut_dptr<T>();

        const int32_t elem_cnt = y_tensor->shape().elem_cnt();
        FOR_RANGE(int32_t, i, 0, elem_cnt){
            dx_ptr[i] = (y_ptr[i] >= static_cast<T>(-3) && y_ptr[i] <= static_cast<T>(3)) ? dy_ptr[i] : static_cast<T>(0);
        }
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override {return false;}
};

#define REGISTER_CPU_HARDSIGMOID_BACKWARD_KERNEL(device, dtype) \
    REGISTER_USER_KERNEL("hardsigmoid_grad")                     \
        .SetCreateFn<CpuHardsigmoidGradKernel<device, dtype>>() \
        .SetIsMatchedHob((HobDeviceTag()==device) & (HobDataType("out", 0) == GetDataType<dtype>::value)) \
        .SetInplaceProposalFn(                                                              \
          [](const InferContext&, AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));              \
            return Maybe<void>::Ok();                                                     \
          });

REGISTER_CPU_HARDSIGMOID_BACKWARD_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_HARDSIGMOID_BACKWARD_KERNEL(DeviceType::kCPU, double)

} // namespace user_op

} // namespace oneflow