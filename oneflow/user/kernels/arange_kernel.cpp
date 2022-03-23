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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"



namespace oneflow {
namespace user_op {
  class ArangeOpKernelCache final : public user_op::OpKernelCache {
 public:
  ArangeOpKernelCache(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~ArangeOpKernelCache() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};
template<DeviceType device_type, typename T>
class ArangeKernel final : public OpKernel {
 public:
  ArangeKernel() = default;
  ~ArangeKernel() = default;
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
    user_op::KernelCacheContext* ctx) const override {
    int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  if (parallel_num > 1) {
        DataType dtype = ctx->Attr<DataType>("dtype");
        int64_t range_elem_cnt = 0;
        if (IsIntegralDataType(dtype)) {
          int64_t integer_delta = ctx->Attr<int64_t>("integer_delta");
          int64_t integer_start = ctx->Attr<int64_t>("integer_start");
          int64_t integer_limit = ctx->Attr<int64_t>("integer_limit");
          range_elem_cnt = std::ceil(static_cast<double>(integer_limit - integer_start) / integer_delta);
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
  view =
      GetTensorSliceView4ParallelId(parallel_hierarchy, nd_sbp, logical_shape, parallel_id);
        return std::make_shared<ArangeOpKernelCache>(view.At(0).begin(),view.At(0).end());
    } else {
      return nullptr;
    }
  }
 private:
  void Compute(user_op::KernelComputeContext* ctx,user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* temp = tmp_buffer->mut_dptr<T>();
    const DataType dtype = ctx->Attr<DataType>("dtype");
    int64_t arange_elem_cnt = 0;
    T start = 0;
    T delta = 0;
    T limit = 0;
    if (IsIntegralDataType(dtype)) {
      start = ctx->Attr<int64_t>("integer_start");
      delta = ctx->Attr<int64_t>("integer_delta");
      limit = ctx->Attr<int64_t>("integer_limit");
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

    ArangeFunctor<device_type, T>()(ctx->stream(), start, delta, arange_elem_cnt, temp);
    int j = 0;
    for (int i = cache->lower(); i < cache->upper(); i++) {
         *(output + j) = *(temp + i); 
         j++;
       }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  private:
      mutable TensorSliceView view;
};

template<typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
  DataType dtype = ctx->Attr<DataType>("dtype");
  int64_t range_elem_cnt = 0;
  if (IsIntegralDataType(dtype)) {
    int64_t integer_delta = ctx->Attr<int64_t>("integer_delta");
    int64_t integer_start = ctx->Attr<int64_t>("integer_start");
    int64_t integer_limit = ctx->Attr<int64_t>("integer_limit");
    range_elem_cnt = std::ceil(static_cast<double>(integer_limit - integer_start) / integer_delta);
  } else {
    double float_delta = ctx->Attr<double>("float_delta");
    double float_start = ctx->Attr<double>("float_start");
    double float_limit = ctx->Attr<double>("float_limit");
    range_elem_cnt = std::ceil(static_cast<double>(float_limit - float_start) / float_delta);
  }
    if (range_elem_cnt == 0) { return; }
    return range_elem_cnt*sizeof(T);
  };
}

#define REGISTER_ARANGE_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("arange").SetCreateFn<ArangeKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                   \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value)).SetInferTmpSizeFn(GenFwInferTmpSizeFn<dtype>());
      
#define REGISTER_ARANGE_KERNELS_WITH_DEVICE(device) \
  REGISTER_ARANGE_KERNEL(device, uint8_t)           \
  REGISTER_ARANGE_KERNEL(device, int8_t)            \
  REGISTER_ARANGE_KERNEL(device, int32_t)           \
  REGISTER_ARANGE_KERNEL(device, int64_t)           \
  REGISTER_ARANGE_KERNEL(device, float)             \
  REGISTER_ARANGE_KERNEL(device, double)

// Register CPU version
REGISTER_ARANGE_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version
#ifdef WITH_CUDA
REGISTER_ARANGE_KERNELS_WITH_DEVICE(DeviceType::kCUDA);
#endif
}  // namespace user_op
}  // namespace oneflow
