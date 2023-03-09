#include "oneflow/user/kernels/fft_kernel_util.h"

namespace oneflow{

template<DeviceType device_type, typename T>
class FftKernel final : public user_op::OpKernel{
public:
    FftKernel() = default;
    ~FftKernel() = default;
private:
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
        const user_op::Tensor* other = ctx->Tensor4ArgNameAndIndex("other", 0);
        user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

        const int64_t elem_num = y->shape_view().elem_cnt();
        const T* x_ptr = x->dptr<T>();
        const T* other_ptr = other->dptr<T>();
        T* y_ptr = y->mut_dptr<T>();

        if (elem_num < GetMaxVal<int32_t>()) {
        FftKernelUtil<device_type, T, int32_t>::FftForward(ctx->stream(), elem_num, 
                                                           x_ptr, other_ptr, y_ptr);
        } else {
        FftKernelUtil<device_type, T, int64_t>::FftForward(ctx->stream(), elem_num, 
                                                           x_ptr, other_ptr, y_ptr);
        }
    }
};


#define REGISTER_FFT_KERNELS(device, dtype)                 \
  REGISTER_USER_KERNEL("fft")                               \
      .SetCreateFn<FftKernel<device, dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value))


REGISTER_FFT_KERNELS(DeviceType::kCPU, float);
REGISTER_FFT_KERNELS(DeviceType::kCPU, double);
REGISTER_FFT_KERNELS(DeviceType::kCPU, uint8_t);
REGISTER_FFT_KERNELS(DeviceType::kCPU, int8_t);
REGISTER_FFT_KERNELS(DeviceType::kCPU, int32_t);
REGISTER_FFT_KERNELS(DeviceType::kCPU, int64_t);

#ifdef WITH_CUDA
REGISTER_FFT_KERNELS(DeviceType::kCUDA, float);
REGISTER_FFT_KERNELS(DeviceType::kCUDA, double);
REGISTER_FFT_KERNELS(DeviceType::kCUDA, uint8_t);
REGISTER_FFT_KERNELS(DeviceType::kCUDA, int8_t);
REGISTER_FFT_KERNELS(DeviceType::kCUDA, int32_t);
REGISTER_FFT_KERNELS(DeviceType::kCUDA, int64_t);
#endif  // WITH_CUDA

}   // namespace oneflow
