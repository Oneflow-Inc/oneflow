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
#include "oneflow/user/kernels/arange_kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {
namespace user_op {
class ArangeOpKernelCache final : public user_op::OpKernelCache {
 public:
  ArangeOpKernelCache(int32_t lower, int32_t upper) : lower_(lower), upper_(upper) {}
  ~ArangeOpKernelCache() override = default;

  int32_t lower() const { return lower_; }
  int32_t upper() const { return upper_; }

 private:
  const int32_t lower_;
  const int32_t upper_;
};
template<DeviceType device_type, typename T>
class ArangeKernel final : public OpKernel, public CudaGraphSupport {
 public:
  ArangeKernel() = default;
  ~ArangeKernel() = default;
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    DataType dtype = ctx->Attr<DataType>("dtype");
    int64_t range_elem_cnt = 0;
    int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    if (parallel_num > 1) {
      if (IsIntegralDataType(dtype)) {
        int64_t integer_delta = ctx->Attr<int64_t>("integer_delta");
        int64_t integer_start = ctx->Attr<int64_t>("integer_start");
        int64_t integer_limit = ctx->Attr<int64_t>("integer_limit");
        range_elem_cnt =
            std::ceil(static_cast<double>(integer_limit - integer_start) / integer_delta);
      } else {
        double float_delta = ctx->Attr<double>("float_delta");
        double float_start = ctx->Attr<double>("float_start");
        double float_limit = ctx->Attr<double>("float_limit");
        range_elem_cnt = std::ceil(static_cast<double>(float_limit - float_start) / float_delta);
      }
      const Shape& logical_shape = Shape({range_elem_cnt});
      const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
      const Shape& parallel_hierarchy = *ctx->parallel_desc().hierarchy();
      const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
      TensorSliceView view =
          GetTensorSliceView4ParallelId(parallel_hierarchy, nd_sbp, logical_shape, parallel_id);
      std::shared_ptr<ArangeOpKernelCache> cache(
          new ArangeOpKernelCache(view.At(0).begin(), view.At(0).end()));
      return cache;
    } else {
      return nullptr;
    }
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    const DataType dtype = ctx->Attr<DataType>("dtype");
    int64_t arange_elem_cnt = 0;
    T start = static_cast<T>(0.0);
    T delta = static_cast<T>(0.0);
    T limit = static_cast<T>(0.0);
    if (IsIntegralDataType(dtype)) {
      start = static_cast<T>(static_cast<double>(ctx->Attr<int64_t>("integer_start")));
      delta = static_cast<T>(static_cast<double>(ctx->Attr<int64_t>("integer_delta")));
      limit = static_cast<T>(static_cast<double>(ctx->Attr<int64_t>("integer_limit")));
      arange_elem_cnt = std::ceil(static_cast<double>(limit - start) / delta);
    } else {
      // If we use static_cast<T>(start, delta, limit) and std::ceil to calculate arange_elem_cnt,
      // it will cause rounding error.
      double float_start = ctx->Attr<double>("float_start");
      double float_delta = ctx->Attr<double>("float_delta");
      double float_limit = ctx->Attr<double>("float_limit");
      arange_elem_cnt = std::ceil(static_cast<double>(float_limit - float_start) / float_delta);
      start = static_cast<T>(float_start);
      delta = static_cast<T>(float_delta);
      limit = static_cast<T>(float_limit);
    }
    if (arange_elem_cnt == 0) { return; }
    if (cache == nullptr) {
      ArangeFunctor<device_type, T>()(ctx->stream(), start, delta, arange_elem_cnt, output);
    } else {
      const auto* arange_cache = dynamic_cast<const ArangeOpKernelCache*>(cache);
      auto arange_len = arange_cache->upper() - arange_cache->lower();
      ArangeFunctor<device_type, T>()(ctx->stream(),
                                      static_cast<T>(start + delta * arange_cache->lower()), delta,
                                      arange_len, output);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ARANGE_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("arange").SetCreateFn<ArangeKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                   \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_ARANGE_KERNELS_WITH_DEVICE(device) \
  REGISTER_ARANGE_KERNEL(device, uint8_t)           \
  REGISTER_ARANGE_KERNEL(device, int8_t)            \
  REGISTER_ARANGE_KERNEL(device, int32_t)           \
  REGISTER_ARANGE_KERNEL(device, int64_t)           \
  REGISTER_ARANGE_KERNEL(device, float)             \
  REGISTER_ARANGE_KERNEL(device, double)

#define REGISTER_ARANGE_KERNELS_WITH_CUDA_HALF(device) REGISTER_ARANGE_KERNEL(device, half)

// Register CPU version
REGISTER_ARANGE_KERNELS_WITH_DEVICE(DeviceType::kCPU);
REGISTER_ARANGE_KERNEL(DeviceType::kCPU, float16);
// Register GPU version
#ifdef WITH_CUDA
REGISTER_ARANGE_KERNELS_WITH_DEVICE(DeviceType::kCUDA);
REGISTER_ARANGE_KERNELS_WITH_CUDA_HALF(DeviceType::kCUDA);
#endif
}  // namespace user_op
}  // namespace oneflow
