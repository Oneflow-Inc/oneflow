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

namespace oneflow {

namespace {
template<typename T>
__global__ void CumSumForwardGpu(const int32_t instance_num, const int32_t instance_size,
                                 const int32_t post, const bool exclusive, const bool reverse,
                                 const T* in, T* out) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    const int32_t start_idx = reverse ? i % post + (i / post + 1) * instance_size * post - post
                                      : i % post + (i / post) * instance_size * post;
    out[start_idx] = 0;
    T temp = 0;
    FOR_RANGE(int32_t, j, exclusive, instance_size) {
      int32_t out_index = reverse ? start_idx - j * post : start_idx + j * post;
      int32_t in_index =
          reverse ? start_idx - (j - exclusive) * post : start_idx + (j - exclusive) * post;
      temp += in[in_index];
      out[out_index] = temp;
    }
  }
}

}  //  namespace

template<typename T>
class GpuCumsumKernel final : public user_op::OpKernel {
 public:
  GpuCumsumKernel() = default;
  ~GpuCumsumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                             in->shape().elem_cnt() * sizeof(T));
    int32_t axis = ctx->Attr<int32_t>("axis");
    const bool exclusive = ctx->Attr<bool>("exclusive");
    const bool reverse = ctx->Attr<bool>("reverse");
    const int32_t elem_cnt = in->shape().elem_cnt();
    int32_t start = exclusive ? 1 : 0;
    int32_t instance_size = in->shape().At(axis);
    int32_t instance_num = elem_cnt / instance_size;
    int32_t post = 1;
    FOR_RANGE(int32_t, i, axis + 1, in->shape().NumAxes()) { post *= in->shape().At(i); }
    RUN_CUDA_KERNEL((CumSumForwardGpu<T>), ctx->device_ctx(), instance_num, instance_num,
                    instance_size, post, exclusive, reverse, in->dptr<T>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_CUMSUM_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<GpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                                \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_CUMSUM_KERNEL(float)
REGISTER_GPU_CUMSUM_KERNEL(double)
REGISTER_GPU_CUMSUM_KERNEL(int32_t)
REGISTER_GPU_CUMSUM_KERNEL(int64_t)

}  // namespace oneflow
