#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ReluDropoutBackwardKernel(const T* dy, const int32_t* mask, T* dx, const int64_t elem_cnt, 
                                            const int64_t cols, const int64_t aux_ld, float scale){
    CUDA_1D_KERNEL_LOOP_T(int64_t, i, elem_cnt){
        const int64_t row = i / cols; 
        const int64_t col = i - row * cols; 
        const int32_t lane_id = col % 32; 
        const int64_t aux_idx = ((row * aux_ld) + col) / 32; 
        bool is_positive = mask[aux_idx] & (1 << lane_id);
        dx[i] = dy[i] * static_cast<T>(is_positive) * static_cast<T>(scale); 
    }
}

template<typename T>
class FusedReluDropoutGradKernel final : public user_op::OpKernel,
                                                    public user_op::CudaGraphSupport {
    public:
    FusedReluDropoutGradKernel() = default;
    ~FusedReluDropoutGradKernel() override = default;

    private:
    using user_op::OpKernel::Compute;
    void Compute(user_op::KernelComputeContext* ctx) const override {

    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float rate = ctx->Attr<float>("rate");
    float scale = 0.0;
    if (rate < 1.0f) { scale = 1.0f / (1.0f - rate); }

    const int64_t cols = dy->shape().At(1); 

    const int64_t aux_ld = mask->shape().At(1); 
    const int64_t elem_cnt = dy->shape().elem_cnt(); 
    ReluDropoutBackwardKernel<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                                        reinterpret_cast<const T*>(dy->dptr()), mask->dptr<int32_t>(), 
                                        reinterpret_cast<T*>(dx->mut_dptr()), elem_cnt, 
                                        cols, aux_ld, scale); 
    }

    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_RELU_DROPOUT_GRAD_KERNEL_GPU(cpp_type, data_type)     \
    REGISTER_USER_KERNEL("fused_relu_dropout_grad")                            \
        .SetCreateFn<FusedReluDropoutGradKernel<cpp_type>>()                     \
        .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                        && (user_op::HobDataType("dx", 0) == data_type));

REGISTER_FUSED_RELU_DROPOUT_GRAD_KERNEL_GPU(float, DataType::kFloat)
    


} // namespace 

} // namespace oneflow 
