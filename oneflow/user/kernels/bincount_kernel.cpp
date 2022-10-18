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
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

template<typename IDX, typename T>
void BinCountComputeWeight(const IDX* in_ptr, const T* weight, T* out_ptr, int64_t size) {
  FOR_RANGE(int64_t, i, 0, size) {
    IDX idx = *(in_ptr + i);
    out_ptr[idx] += weight[i];
  }
}

template<typename IDX, typename T>
void BinCountCompute(const IDX* in_ptr, T* out_ptr, int64_t size) {
  FOR_RANGE(int64_t, i, 0, size) {
    IDX idx = *(in_ptr + i);
    out_ptr[idx] += 1L;
  }
}

template<typename IDX, typename T>
class CpuBinCountKernel final : public user_op::OpKernel {
 public:
  CpuBinCountKernel() = default;
  ~CpuBinCountKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    size_t out_size = ctx->Attr<int64_t>("size") * sizeof(T);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const IDX* in_ptr = in->dptr<IDX>();
    T* out_ptr = out->mut_dptr<T>();
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), out_ptr, 0, out_size);
    int64_t in_size = in->shape_view().elem_cnt();
    if (ctx->has_input("weight", 0)) {
      const T* weight_ptr = ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>();
      BinCountComputeWeight<IDX, T>(in_ptr, weight_ptr, out_ptr, in_size);
    } else {
      BinCountCompute<IDX, T>(in_ptr, out_ptr, in_size);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_BINCOUNT_KERNEL(idx_type, dtype)                                     \
  REGISTER_USER_KERNEL("bincount")                                                        \
      .SetCreateFn<CpuBinCountKernel<idx_type, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                     \
                       && (user_op::HobDataType("in", 0) == GetDataType<idx_type>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_BINCOUNT_KERNEL(int64_t, int64_t)
REGISTER_CPU_BINCOUNT_KERNEL(int64_t, float16)
REGISTER_CPU_BINCOUNT_KERNEL(int64_t, float)
REGISTER_CPU_BINCOUNT_KERNEL(int64_t, double)

}  // namespace oneflow
