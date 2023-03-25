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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/ep/primitive/cast.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

namespace {

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

}  // namespace

template<typename T>
class MluArangeKernel final : public user_op::OpKernel {
 public:
  MluArangeKernel() = default;
  ~MluArangeKernel() = default;

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
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType dtype = ctx->Attr<DataType>("dtype");
    int32_t start_int = 0;
    int32_t step_int = 0;
    float start_float = 0.0;
    float step_float = 0.0;

    DataType tmp_out_data_type;
    size_t tmp_out_workspace_size = out->shape_view().elem_cnt();
    if (IsIntegralDataType(dtype)) {
      tmp_out_workspace_size *= GetSizeOfDataType(DataType::kInt32);
      tmp_out_data_type = DataType::kInt32;
    } else if (dtype == DataType::kFloat16) {
      tmp_out_workspace_size *= GetSizeOfDataType(DataType::kFloat16);
      tmp_out_data_type = DataType::kFloat16;
    } else {
      tmp_out_workspace_size *= GetSizeOfDataType(kFloat);
      tmp_out_data_type = DataType::kFloat;
    }
    CnnlTensorDescriptor tmp_out_desc;
    tmp_out_desc.set(out, ConvertToCnnlDataType(tmp_out_data_type));

    void* tmp_out_ptr = out->mut_dptr<T>();
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), 0);
    if (tmp_out_data_type != dtype) {
      tmp_out_cnnl_workspace.resize(tmp_out_workspace_size);
      tmp_out_ptr = tmp_out_cnnl_workspace.dptr();
    }

    if (IsIntegralDataType(GetDataType<T>::value)) {
      start_int = static_cast<int32_t>(ctx->Attr<int64_t>("integer_start"));
      step_int = static_cast<int32_t>(ctx->Attr<int64_t>("integer_delta"));
      if (cache) {
        const auto* arange_cache = dynamic_cast<const ArangeOpKernelCache*>(cache);
        start_int += step_int * arange_cache->lower();
      }
      OF_CNNL_CHECK(cnnlArange_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                  CNNL_COMPUTATION_HIGH_PRECISION, (void*)&start_int,
                                  (void*)&step_int, tmp_out_desc.desc(), tmp_out_ptr));
    } else {
      start_float = static_cast<float>(ctx->Attr<double>("float_start"));
      step_float = static_cast<float>(ctx->Attr<double>("float_delta"));
      if (cache) {
        const auto* arange_cache = dynamic_cast<const ArangeOpKernelCache*>(cache);
        start_int += step_int * arange_cache->lower();
      }
      OF_CNNL_CHECK(cnnlArange_v2(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                  CNNL_COMPUTATION_HIGH_PRECISION, (void*)&start_float,
                                  (void*)&step_float, tmp_out_desc.desc(), tmp_out_ptr));
    }

    if (tmp_out_data_type != dtype) {
      CnnlTensorDescriptor out_dec(out);
      cnnlCastDataType_t type = ep::primitive::GetCnnlCastType(tmp_out_data_type, dtype);
      OF_CNNL_CHECK(cnnlCastDataType(ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
                                     tmp_out_desc.desc(), tmp_out_ptr, type, out_dec.desc(),
                                     out->mut_dptr<T>()));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ARANGE_MLU_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("arange").SetCreateFn<MluArangeKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                    \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_ARANGE_MLU_KERNEL(float)
REGISTER_ARANGE_MLU_KERNEL(float16)
REGISTER_ARANGE_MLU_KERNEL(int8_t)
REGISTER_ARANGE_MLU_KERNEL(uint8_t)
REGISTER_ARANGE_MLU_KERNEL(int32_t)
REGISTER_ARANGE_MLU_KERNEL(uint32_t)
REGISTER_ARANGE_MLU_KERNEL(int64_t)
REGISTER_ARANGE_MLU_KERNEL(uint64_t)

}  // namespace oneflow
