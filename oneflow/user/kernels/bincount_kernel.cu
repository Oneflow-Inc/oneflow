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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {
namespace {

template<typename IDX, typename T, bool has_weight>
struct BinCountCompute {
  OF_DEVICE_FUNC static void Compute(const IDX* in_ptr, const T* weight, T* out_ptr, int64_t size) {
  }
};

template<typename IDX, typename T>
struct BinCountCompute<IDX, T, true> {
  OF_DEVICE_FUNC static void Compute(const IDX* in_ptr, const T* weight, T* out_ptr, int64_t size) {
    CUDA_1D_KERNEL_LOOP(i, size) {
      IDX idx = *(in_ptr + i);
      out_ptr[idx] += weight[i];
    }
  }
};

template<typename IDX, typename T>
struct BinCountCompute<IDX, T, false> {
  OF_DEVICE_FUNC static void Compute(const IDX* in_ptr, const T* weight, T* out_ptr, int64_t size) {
    CUDA_1D_KERNEL_LOOP(i, size) {
      IDX idx = *(in_ptr + i);
      out_ptr[idx] += 1L;
    }
  }
};

template<typename IDX, typename T>
class CUDABinCountKernel final : public user_op::OpKernel {
 public:
  CUDABinCountKernel() = default;
  ~CUDABinCountKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    // int64_t size = in->shape_view().elem_cnt();
    // user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    // const IDX* in_ptr = in->dptr<IDX>();
    // T* out_ptr = out->mut_dptr<T>();
    // Memset<DeviceType::kCUDA>(ctx->stream(), out_ptr, 0, size * sizeof(int64_t));
    // if (ctx->has_input("weight", 0)) {
    //   const T* weight_ptr = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
    //   BinCountCompute<IDX, T, true>::Compute(in_ptr, weight_ptr, out_ptr, size);
    // } else {
    //   BinCountCompute<IDX, T, false>::Compute(in_ptr, nullptr, out_ptr, size);
    // }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_CUDA_BINCOUNT_KERNEL(idx_type, dtype)                                    \
  REGISTER_USER_KERNEL("bincount")                                                        \
      .SetCreateFn<CUDABinCountKernel<idx_type, dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                    \
                       && (user_op::HobDataType("in", 0) == GetDataType<idx_type>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, int64_t)
REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, float)
REGISTER_CUDA_BINCOUNT_KERNEL(int64_t, double)


}  // namespace user_op
}  // namespace oneflow
