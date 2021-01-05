#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {
namespace {
template <DeviceType device_type, typename T>   
class DiagKernel final : public user_op::OpKernel {
    public:
        DiagKernel() = default;
        ~DiagKernel() = default;
    private:
        void Compute(user_op::KernelComputeContext *ctx) const override {
            const int32_t dimension = ctx->Attr<int32_t>("dimension");
            const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
            user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("diag_out", 0);
            const ShapeView& out_shape = out_tensor->shape();
            const ShapeView& in_shape = in_tensor->shape();
            int32_t in_dim = in_shape.NumAxes();
            if (in_dim == 1) {
                int32_t stride_0 = out_shape.At(1);
                int32_t stride_1 = 1;
                int32_t in_stride = 1;
                //const TensorBuffer& out_data = out_tensor->dptr<TensorBuffer>();
                //auto* in_data = in_tensor->mut_dptr<unsigned char>();
                //const TensorBuffer& in_data = in_tensor->dptr<TensorBuffer>();
                
                out_tensor += (dimension >= 0 ? dimension*stride_1 : -dimension*stride_0);

                for (int32_t i = 0; i < in_dim; i++) {
                    out_tensor[i * (stride_0 + stride_1)] = in_tensor[i * in_stride];
                }
            } else {
                int32_t stride_0 = in_shape.At(1);
                int32_t stride_1 = 1;
                int32_t out_stride = 1;
                //auto out_data = out_tensor->mut_dptr;
                //auto in_data = in_tensor->mut_dptr;
                int32_t sz = 9;
 
                out_tensor += (dimension >= 0 ? dimension*stride_1 : -dimension*stride_0);
                if (dimension >= 0) {
                        sz = std::min(in_shape.At(0), in_shape.At(1) - dimension);
                    } else {
                        sz = std::min(in_shape.At(0) + dimension, in_shape.At(1));
                    }
                for (int32_t i = 0; i < sz; i++) {
                    out_tensor[i * out_stride] = in_tensor[i * (stride_0 + stride_1)];
                    }
            }


         }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
} // namespace

#define REGISTER_DIAG_KERNEL(device, dtype)                                     \
    REGISTER_USER_KERNEL("diag")                                                \
        .SetCreateFn<DiagKernel<device, dtype>>()                               \
        .SetIsMatchedHob((user_op::HobDeviceTag() == device)                     \
        & (user_op::HobDataType("diag_out", 0) == GetDataType<dtype>::value));

#define REGISTER_DIAG_KERNEL_WITH_DEVICE(device) \
        REGISTER_DIAG_KERNEL(device, float)            \
        REGISTER_DIAG_KERNEL(device, double)           \
        REGISTER_DIAG_KERNEL(device, int8_t)           \
        REGISTER_DIAG_KERNEL(device, int32_t)          \
        REGISTER_DIAG_KERNEL(device, int64_t)

REGISTER_DIAG_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_DIAG_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_DIAG_KERNEL(DeviceType::kGPU, float16)
#endif

} // namespace oneflow