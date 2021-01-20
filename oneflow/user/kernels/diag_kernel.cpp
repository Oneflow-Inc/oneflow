/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"

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

            Memset<device_type>(ctx->device_ctx(), out_tensor->mut_dptr(), 0,
                        out_shape.elem_cnt() * sizeof(T));


            const T* in_buf =  in_tensor->dptr<T>();
            T* out_buf =  out_tensor->mut_dptr<T>();

            if (in_dim == 1) {
                int32_t stride_0 = out_shape.At(1);
                int32_t stride_1 = 1;
                int32_t input_cnt = in_shape.elem_cnt();
                out_buf += (dimension >= 0 ? dimension*stride_1 : -dimension*stride_0);
                for (int32_t i = 0; i < input_cnt; i++) {
                    out_buf[i * (stride_0 + stride_1)] = in_buf[i];
                }
            } else {
                int32_t stride_0 = in_shape.At(1);
                int32_t stride_1 = 1;
                int32_t sz = 0;
 
                in_buf += (dimension >= 0 ? dimension*stride_1 : -dimension*stride_0);
                if (dimension >= 0) {
                        sz = std::min(in_shape.At(0), in_shape.At(1) - dimension);
                    } else {
                        sz = std::min(in_shape.At(0) + dimension, in_shape.At(1));
                    }
                for (int32_t i = 0; i < sz; i++) {
                    out_buf[i] = in_buf[i * (stride_0 + stride_1)];
                    }
            }
         }
  
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class DiagGradKernel final : public user_op::OpKernel {
 public:
  DiagGradKernel() = default;
  ~DiagGradKernel() = default;

  private:
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
        user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
        int32_t dimension = ctx->Attr<int32_t>("dimension");
        const ShapeView& dx_shape = dx->shape();
        const ShapeView& dy_shape = dy->shape();
        int32_t in_dim = dx_shape.NumAxes();
        int32_t dy_num_cnt = dy_shape.At(0);
        int32_t dx_num_cnt = dx_shape.Count(0);
        T* dx_buf =  dx->mut_dptr<T>();
        const T* dy_buf = dy->dptr<T>();

        Memset<DeviceType::kCPU>(ctx->device_ctx(), dx->mut_dptr<T>(), 0,
                             dx_shape.elem_cnt() * sizeof(T));

        if (in_dim == 1) {
            int32_t stride_1 = 1;
            int32_t stride_0 = dy_shape.At(1);
            
            dy_buf += (dimension >= 0 ? dimension*stride_1 : -dimension*stride_0);
            for (int32_t i = 0; i < dx_num_cnt; i++) {
                    dx_buf[i] = dy_buf[i *  (stride_0 + stride_1)];
            }
        } else {
                int32_t stride_0 = dx_shape.At(1);
                int32_t stride_1 = 1;
                dx_buf += (dimension >= 0 ? dimension*stride_1 : -dimension*stride_0);
                for (int32_t i = 0; i < dy_num_cnt; i++) {
                    dx_buf[i * (stride_0 + stride_1)] = dy_buf[i];
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
        & (user_op::HobDataType("input_tensor", 0) == GetDataType<dtype>::value));

#define REGISTER_DIAG_GRAD_KERNEL(device, dtype)                                     \
    REGISTER_USER_KERNEL("diag_grad")                                                \
        .SetCreateFn<DiagGradKernel<device, dtype>>()                               \
        .SetIsMatchedHob((user_op::HobDeviceTag() == device)                     \
        & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_DIAG_KERNEL_WITH_DEVICE(device) \
        REGISTER_DIAG_KERNEL(device, float)            \
        REGISTER_DIAG_KERNEL(device, double)           \
        REGISTER_DIAG_KERNEL(device, int8_t)           \
        REGISTER_DIAG_KERNEL(device, int32_t)          \
        REGISTER_DIAG_KERNEL(device, int64_t)           \
        REGISTER_DIAG_GRAD_KERNEL(device, float)            \
        REGISTER_DIAG_GRAD_KERNEL(device, double)           \
        REGISTER_DIAG_GRAD_KERNEL(device, int8_t)           \
        REGISTER_DIAG_GRAD_KERNEL(device, int32_t)          \
        REGISTER_DIAG_GRAD_KERNEL(device, int64_t)           

REGISTER_DIAG_KERNEL_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_DIAG_KERNEL_WITH_DEVICE(DeviceType::kGPU)
REGISTER_DIAG_KERNEL(DeviceType::kGPU, float16)
#endif

} // namespace oneflow