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
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"

namespace oneflow {

namespace {

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewReduceMatmulTransAPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("y", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/true,
                            /*transpose_b=*/false);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewReduceMatmulNoTransAPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("y", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/false,
                            /*transpose_b=*/false);
}

auto ReduceMatmulTransAPrimitiveExists() {
  return hob::make_custom("ReduceMatmulTransAPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewReduceMatmulTransAPrimitive(&ctx).operator bool();
                          });
}

auto ReduceMatmulNoTransAPrimitiveExists() {
  return hob::make_custom("ReduceMatmulNoTransAPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewReduceMatmulNoTransAPrimitive(&ctx).operator bool();
                          });
}

size_t ReduceSumLikeInferTmpSize(user_op::InferContext* ctx) {
  if (ctx->Attr<std::vector<int32_t>>("axis").empty()) { return 0; }
  const user_op::TensorDesc& tensor_desc_x = ctx->InputTensorDesc("x", 0);
  return tensor_desc_x.shape().elem_cnt() * GetSizeOfDataType(tensor_desc_x.data_type());
}

}  // namespace

template<DeviceType device_type, typename T>
class ReduceSumLikeOpKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ReduceSumLikeOpKernel() = default;
  ~ReduceSumLikeOpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
    if (tensor_x->shape_view().elem_cnt() == 0) {
      if (tensor_y->shape_view().elem_cnt() != 0) {
        Memset<device_type>(
            ctx->stream(), tensor_y->mut_dptr<T>(), 0,
            tensor_y->shape_view().elem_cnt() * GetSizeOfDataType(tensor_y->data_type()));
      }
      return;
    }
    if (axis.empty()) {
      CHECK_EQ(tensor_x->shape_view(), tensor_y->shape_view());
      Memcpy<device_type>(
          ctx->stream(), tensor_y->mut_dptr(), tensor_x->dptr(),
          tensor_x->shape_view().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    } else {
      user_op::Tensor* tensor_tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      T* temp_storage = static_cast<T*>(tensor_tmp->mut_dptr());
      NdarrayUtil<device_type, T>::ReduceSum(
          ctx->stream(),
          XpuVarNdarray<T>(CreateReducedShape(tensor_x->shape_view(), {axis.begin(), axis.end()}),
                           tensor_y->mut_dptr<T>()),
          XpuVarNdarray<const T>(tensor_x->shape_view(), tensor_x->dptr<T>(),
                                 tensor_x->shape_view().NumAxes()),
          XpuVarNdarray<T>(tensor_x->shape_view(), temp_storage, tensor_x->shape_view().NumAxes()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_SUM_LIKE_KERNEL(device, data_type_pair)                                \
  REGISTER_USER_KERNEL("reduce_sum_like")                                                      \
      .SetCreateFn<ReduceSumLikeOpKernel<device, OF_PP_PAIR_FIRST(data_type_pair)>>()          \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                    \
                       && (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(data_type_pair))) \
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

class ReduceSumLikeHalfKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ReduceSumLikeHalfKernel() = default;
  ~ReduceSumLikeHalfKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<int32_t> axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));
    const user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (axis.empty()) {
      CHECK_EQ(tensor_x->shape_view(), tensor_y->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), tensor_y->mut_dptr(), tensor_x->dptr(),
          tensor_x->shape_view().elem_cnt() * GetSizeOfDataType(tensor_x->data_type()));
    } else {
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const ShapeView& in_shape = tensor_x->shape_view();
      bool is_axis_contiguous = false;
      int64_t outer_size = 0, inner_size = 0, reduce_size = 0;
      GetReduceSumLayout(axis, in_shape, &is_axis_contiguous, &outer_size, &inner_size,
                         &reduce_size);
      if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {
        bool trans_a = (inner_size != 1);
        const int32_t m = (inner_size == 1) ? outer_size : inner_size;
        const int32_t n = 1;
        const int32_t k = reduce_size;
        std::unique_ptr<ep::primitive::Fill> fill =
            ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                                    tensor_x->data_type());
        CHECK(fill);
        fill->Launch(ctx->stream(), tmp_buffer->mut_dptr(), 1.0, reduce_size);

        std::unique_ptr<ep::primitive::Matmul> matmul;
        if (trans_a) {
          matmul = NewReduceMatmulTransAPrimitive(ctx);
        } else {
          matmul = NewReduceMatmulNoTransAPrimitive(ctx);
        }
        CHECK(matmul);
        matmul->Launch(ctx->stream(), m, n, k, 1.0, tensor_x->dptr(), tmp_buffer->dptr(), 0.0,
                       tensor_y->mut_dptr());

      } else {
        const Shape& reduced_shape = CreateReducedShape(in_shape, {axis.begin(), axis.end()});
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
                 tmp_buffer->shape_view().elem_cnt());
        auto h2f = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
            ctx->device_type(), tensor_x->data_type(), DataType::kFloat);
        CHECK(h2f);
        auto f2h = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
            ctx->device_type(), DataType::kFloat, tensor_x->data_type());
        CHECK(f2h);
        h2f->Launch(ctx->stream(), tensor_x->dptr(), in_tmp_buffer, in_shape.elem_cnt());

        NdarrayReduce<DeviceType::kCUDA, float, BinaryFuncSum>::Reduce(
            ctx->stream(), XpuVarNdarray<float>(reduced_shape, out_tmp_buffer),
            XpuVarNdarray<const float>(in_shape, in_tmp_buffer),
            XpuVarNdarray<float>(in_shape, reduce_tmp_buffer));

        f2h->Launch(ctx->stream(), out_tmp_buffer, tensor_y->mut_dptr(),
                    tensor_y->shape_view().elem_cnt());
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REDUCE_SUM_LIKE_HALF_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("reduce_sum_like")                                               \
      .SetCreateFn<ReduceSumLikeHalfKernel>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)   \
                       && ReduceMatmulTransAPrimitiveExists()                           \
                       && ReduceMatmulNoTransAPrimitiveExists())                        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const Shape& in_shape = ctx->InputTensorDesc("x", 0).shape();                   \
        const Shape& out_shape = ctx->OutputTensorDesc("y", 0).shape();                 \
        const auto& axis = RegularAxis(ctx->Attr<std::vector<int32_t>>("axis"));        \
        if (axis.empty()) {                                                             \
          size_t tmp_bytes = 0;                                                         \
          return tmp_bytes;                                                             \
        }                                                                               \
        bool is_axis_contiguous = false;                                                \
        int64_t outer_size = 0, inner_size = 0, reduce_size = 0;                        \
        GetReduceSumLayout(axis, ShapeView(in_shape), &is_axis_contiguous, &outer_size, \
                           &inner_size, &reduce_size);                                  \
        size_t tmp_bytes = 0;                                                           \
        if (is_axis_contiguous && (outer_size == 1 || inner_size == 1)) {               \
          tmp_bytes = GetCudaAlignedSize(reduce_size * sizeof(dtype));                  \
        } else {                                                                        \
          tmp_bytes = (2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float))      \
                       + GetCudaAlignedSize(out_shape.elem_cnt() * sizeof(float)));     \
        }                                                                               \
        return tmp_bytes;                                                               \
      });
REGISTER_REDUCE_SUM_LIKE_HALF_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_REDUCE_SUM_LIKE_HALF_KERNEL(nv_bfloat16)
#endif

#endif

}  // namespace oneflow
