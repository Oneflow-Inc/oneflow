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
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

size_t ReduceSumLikeInferTmpSize(user_op::InferContext* ctx) {
  if (ctx->Attr<std::vector<int32_t>>("axis").empty()) { return 0; }
  const user_op::TensorDesc* tensor_desc_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  return tensor_desc_x->shape().elem_cnt() * GetSizeOfDataType(tensor_desc_x->data_type());
}

}  // namespace

template<DeviceType device_type, typename T>
class ReduceSumLikeOpKernel final : public user_op::OpKernel {
 public:
  ReduceSumLikeOpKernel() = default;
  ~ReduceSumLikeOpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    if (axis.empty()) {
      CHECK_EQ(tensor_x->shape(), tensor_y->shape());
      Memcpy<device_type>(ctx->device_ctx(), tensor_y->mut_dptr(), tensor_x->dptr(),
                          tensor_x->shape().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    } else {
      user_op::Tensor* tensor_tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      T* temp_storage = static_cast<T*>(tensor_tmp->mut_dptr());
      NdarrayUtil<device_type, T>::ReduceSum(
          ctx->device_ctx(),
          XpuVarNdarray<T>(CreateReducedShape(tensor_x->shape(), {axis.begin(), axis.end()}),
                           tensor_y->mut_dptr<T>()),
          XpuVarNdarray<const T>(tensor_x->shape(), tensor_x->dptr<T>(),
                                 tensor_x->shape().NumAxes()),
          XpuVarNdarray<T>(tensor_x->shape(), temp_storage, tensor_x->shape().NumAxes()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_SUM_LIKE_KERNEL(device, data_type_pair)                               \
  REGISTER_USER_KERNEL("reduce_sum_like")                                                     \
      .SetCreateFn<ReduceSumLikeOpKernel<device, OF_PP_PAIR_FIRST(data_type_pair)>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                    \
                       & (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(data_type_pair))) \
      .SetInferTmpSizeFn(ReduceSumLikeInferTmpSize);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_REDUCE_SUM_LIKE_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

#if defined(WITH_CUDA)

namespace {

std::vector<int32_t> RegularAxis(const std::vector<int32_t>& axis) {
  std::vector<int32_t> regular_axis = axis;
  std::sort(regular_axis.begin(), regular_axis.end());
  return regular_axis;
}

void GetReduceSumLayout(const std::vector<int32_t>& axis, const ShapeView& in_shape,
                        bool* is_axis_contiguous, int64_t* outer_size, int64_t* inner_size,
                        int64_t* reduce_size) {
  *is_axis_contiguous = ((axis.back() - axis.front() + 1) == axis.size());
  *outer_size = in_shape.Count(0, axis.front());
  *inner_size = in_shape.Count(axis.back() + 1);
  *reduce_size = in_shape.Count(axis.front(), axis.back() + 1);
}

}  // namespace

class ReduceSumLikeHalfKernel final : public user_op::OpKernel {
 public:
  explicit ReduceSumLikeHalfKernel(user_op::KernelCreateContext* ctx) {
    axis_ = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
  }
  ~ReduceSumLikeHalfKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (axis_.empty()) {
      CHECK_EQ(tensor_x->shape(), tensor_y->shape());
      Memcpy<DeviceType::kGPU>(
          ctx->device_ctx(), tensor_y->mut_dptr(), tensor_x->dptr(),
          tensor_x->shape().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    } else {
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const ShapeView& in_shape = tensor_x->shape();
      bool is_axis_contiguous = false;
      int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
      GetReduceSumLayout(axis_, in_shape, &is_axis_contiguous, &outer_size, &inner_size,
                         &reduce_size);
      if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {
        CBLAS_TRANSPOSE trans_a = (inner_size == 1) ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE trans_b = CblasNoTrans;
        const int32_t m = (inner_size == 1) ? outer_size : inner_size;
        const int32_t n = 1;
        const int32_t k = reduce_size;
        NewKernelUtil<DeviceType::kGPU>::Fill(ctx->device_ctx(), reduce_size,
                                              static_cast<float16>(1.0),
                                              tmp_buffer->mut_dptr<float16>());
        NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), trans_a, trans_b, m, n, k,
                                                GetOneVal<float16>(), tensor_x->dptr<float16>(),
                                                tmp_buffer->dptr<float16>(), GetZeroVal<float16>(),
                                                tensor_y->mut_dptr<float16>());
      } else {
        const Shape& reduced_shape = CreateReducedShape(in_shape, {axis_.begin(), axis_.end()});
        float* in_tmp_buffer = tmp_buffer->mut_dptr<float>();
        const size_t in_tmp_buffer_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
        float* out_tmp_buffer =
            reinterpret_cast<float*>(tmp_buffer->mut_dptr<char>() + in_tmp_buffer_bytes);
        const size_t out_tmp_buffer_bytes =
            GetCudaAlignedSize(reduced_shape.elem_cnt() * sizeof(float));
        float* reduce_tmp_buffer = reinterpret_cast<float*>(
            tmp_buffer->mut_dptr<char>() + in_tmp_buffer_bytes + out_tmp_buffer_bytes);
        const size_t reduce_tmp_buffer_bytes =
            GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
        CHECK_LE(in_tmp_buffer_bytes + out_tmp_buffer_bytes + reduce_tmp_buffer_bytes,
                 tmp_buffer->shape().elem_cnt());
        CopyElemOnGpu<float16, float>(ctx->device_ctx(), tensor_x->dptr<float16>(), in_tmp_buffer,
                                      in_shape.elem_cnt());

        NdarrayReduce<DeviceType::kGPU, float, BinaryFuncSum>::Reduce(
            ctx->device_ctx(), XpuVarNdarray<float>(reduced_shape, out_tmp_buffer),
            XpuVarNdarray<const float>(in_shape, in_tmp_buffer),
            XpuVarNdarray<float>(in_shape, reduce_tmp_buffer));

        CopyElemOnGpu<float, float16>(ctx->device_ctx(), out_tmp_buffer,
                                      tensor_y->mut_dptr<float16>(), tensor_y->shape().elem_cnt());
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  std::vector<int32_t> axis_;
};

REGISTER_USER_KERNEL("reduce_sum_like")
    .SetCreateWithCtxFn<ReduceSumLikeHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("y", 0) == GetDataType<float16>::value))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const Shape& in_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
      const Shape& out_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();
      const auto& axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
      if (axis.empty()) {
        size_t tmp_bytes = 0;
        return tmp_bytes;
      }
      bool is_axis_contiguous = false;
      int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
      GetReduceSumLayout(axis, ShapeView(in_shape), &is_axis_contiguous, &outer_size, &inner_size,
                         &reduce_size);
      size_t tmp_bytes = 0;
      if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {
        tmp_bytes = GetCudaAlignedSize(reduce_size * sizeof(float16));
      } else {
        tmp_bytes = (2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float))
                     + GetCudaAlignedSize(out_shape.elem_cnt() * sizeof(float)));
      }
      return tmp_bytes;
    });

#endif

}  // namespace oneflow
