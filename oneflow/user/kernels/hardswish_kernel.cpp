# include "oneflow/core/framework/framework.h"
# include "oneflow/core/common/data_type.h"

namespace oneflow{

namespace user_op{

template<DeviceType device_type, typename T>
class CpuHardSwishKernel final : public OpKernel {
public: 
    CpuHardSwishKernel() = default;
    ~CpuHardSwishKernel() = default;
private: 
    void Compute(KernelComputeContext* ctx) const override {
        const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
        Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
        const T* in_ptr = in_tensor->dptr<T>();
        T* out_ptr = out_tensor->mut_dptr<T>();

        const int32_t elem_cnt = in_tensor->shape().elem_cnt();
        FOR_RANGE(int32_t, i, 0, elem_cnt){
            if(in_ptr[i] <= static_cast<T>(-3)){
                out_ptr[i] = 0;
            }else if(in_ptr[i] >= static_cast<T>(3)){
                out_ptr[i] = in_ptr[i];
            }
            else{
                out_ptr[i] = (in_ptr[i]*(in_ptr[i]+static_cast<T>(3))) / static_cast<T>(6);
            }
        }
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override {return false; }
};

#define REGISTER_CPU_HARDSWISH_KERNEL(device, dtype) \
    REGISTER_USER_KERNEL("hardswish") \
        .SetCreateFn<CpuHardSwishKernel<device, dtype>>() \ 
        .SetIsMatchedHob((HobDeviceTag() == device) \
                            & (HobDataType("out", 0) == GetDataType<dtype>::value));


REGISTER_CPU_HARDSWISH_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_HARDSWISH_KERNEL(DeviceType::kCPU, double)

} // namespace user_op 

} // namespace oneflow
