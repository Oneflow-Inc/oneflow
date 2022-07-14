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
#include "oneflow/user/kernels/where_kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
class WhereKernel final : public user_op::OpKernel {
 public:
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (!(x->shape_view() == y->shape_view() && y->shape_view() == cond->shape_view())) {
      size_t num_axes = out->shape_view().NumAxes();
      int64_t elem_cnt = out->shape_view().elem_cnt();
      const size_t x_bytes = GetCudaAlignedSize(elem_cnt * sizeof(T));
      const size_t y_bytes = GetCudaAlignedSize(elem_cnt * sizeof(T));
      T* y_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + x_bytes);
      CondT* cond_tmp_buf =
          reinterpret_cast<CondT*>(tmp_buffer->mut_dptr<char>() + x_bytes + y_bytes);
      NdarrayUtil<device_type, T>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<T>(out->shape_view(), tmp_buffer->mut_dptr<T>()),
          XpuVarNdarray<const T>(x->shape_view(), x->dptr<T>(), num_axes));
      NdarrayUtil<device_type, T>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<T>(out->shape_view(), y_tmp_buf),
          XpuVarNdarray<const T>(y->shape_view(), y->dptr<T>(), num_axes));
      NdarrayUtil<device_type, CondT>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<CondT>(out->shape_view(), cond_tmp_buf),
          XpuVarNdarray<const CondT>(cond->shape_view(), cond->dptr<CondT>(), num_axes));
      WhereKernelUtil<device_type, T, CondT>::Where(ctx->stream(), out->shape_view().elem_cnt(),
                                                    cond_tmp_buf, tmp_buffer->mut_dptr<T>(),
                                                    y_tmp_buf, out->mut_dptr<T>());
    } else {
      WhereKernelUtil<device_type, T, CondT>::Where(ctx->stream(), out->shape_view().elem_cnt(),
                                                    cond->dptr<CondT>(), x->dptr<T>(), y->dptr<T>(),
                                                    out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename CondT>
class WhereScalarXKernel final : public user_op::OpKernel {
 public:
  WhereScalarXKernel() = default;
  ~WhereScalarXKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else if (ctx->Attr<bool>("has_bool_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<bool>("bool_operand"));
    } else {
      UNIMPLEMENTED() << "The scalar in Where should be bool, float or int.";
    }
    if (!(y->shape_view() == cond->shape_view())) {
      size_t num_axes = out->shape_view().NumAxes();
      int64_t elem_cnt = out->shape_view().elem_cnt();
      const size_t y_bytes = GetCudaAlignedSize(elem_cnt * sizeof(T));
      CondT* cond_tmp_buf = reinterpret_cast<CondT*>(tmp_buffer->mut_dptr<char>() + y_bytes);
      NdarrayUtil<device_type, T>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<T>(out->shape_view(), tmp_buffer->mut_dptr<T>()),
          XpuVarNdarray<const T>(y->shape_view(), y->dptr<T>(), num_axes));
      NdarrayUtil<device_type, CondT>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<CondT>(out->shape_view(), cond_tmp_buf),
          XpuVarNdarray<const CondT>(cond->shape_view(), cond->dptr<CondT>(), num_axes));
      WhereKernelUtil<device_type, T, CondT>::WhereXScalar(
          ctx->stream(), out->shape_view().elem_cnt(), cond_tmp_buf, scalar_operand,
          tmp_buffer->mut_dptr<T>(), out->mut_dptr<T>());
    } else {
      WhereKernelUtil<device_type, T, CondT>::WhereXScalar(
          ctx->stream(), out->shape_view().elem_cnt(), cond->dptr<CondT>(), scalar_operand,
          y->dptr<T>(), out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename CondT>
class WhereScalarYKernel final : public user_op::OpKernel {
 public:
  WhereScalarYKernel() = default;
  ~WhereScalarYKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else if (ctx->Attr<bool>("has_bool_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<bool>("bool_operand"));
    } else {
      UNIMPLEMENTED() << "The scalar in Where should be bool, float or int";
    }
    if (!(x->shape_view() == cond->shape_view())) {
      size_t num_axes = out->shape_view().NumAxes();
      int64_t elem_cnt = out->shape_view().elem_cnt();
      const size_t x_bytes = GetCudaAlignedSize(elem_cnt * sizeof(T));
      CondT* cond_tmp_buf = reinterpret_cast<CondT*>(tmp_buffer->mut_dptr<char>() + x_bytes);
      NdarrayUtil<device_type, T>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<T>(out->shape_view(), tmp_buffer->mut_dptr<T>()),
          XpuVarNdarray<const T>(x->shape_view(), x->dptr<T>(), num_axes));
      NdarrayUtil<device_type, CondT>::BroadcastTo(
          ctx->stream(), XpuVarNdarray<CondT>(out->shape_view(), cond_tmp_buf),
          XpuVarNdarray<const CondT>(cond->shape_view(), cond->dptr<CondT>(), num_axes));
      WhereKernelUtil<device_type, T, CondT>::WhereYScalar(
          ctx->stream(), out->shape_view().elem_cnt(), cond_tmp_buf, tmp_buffer->mut_dptr<T>(),
          scalar_operand, out->mut_dptr<T>());
    } else {
      WhereKernelUtil<device_type, T, CondT>::WhereYScalar(
          ctx->stream(), out->shape_view().elem_cnt(), cond->dptr<CondT>(), x->dptr<T>(),
          scalar_operand, out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename CondT>
class WhereScalarXYKernel final : public user_op::OpKernel {
 public:
  WhereScalarXYKernel() = default;
  ~WhereScalarXYKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (out->shape_view().elem_cnt() == 0) { return; }
    T x_scalar_operand = static_cast<T>(0);
    T y_scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_x_int_operand") && ctx->Attr<bool>("has_y_int_operand")) {
      x_scalar_operand = static_cast<T>(ctx->Attr<int64_t>("x_int_operand"));
      y_scalar_operand = static_cast<T>(ctx->Attr<int64_t>("y_int_operand"));
    } else if (ctx->Attr<bool>("has_x_float_operand") && ctx->Attr<bool>("has_y_float_operand")) {
      x_scalar_operand = static_cast<T>(ctx->Attr<double>("x_float_operand"));
      y_scalar_operand = static_cast<T>(ctx->Attr<double>("y_float_operand"));
    } else if (ctx->Attr<bool>("has_x_bool_operand") && ctx->Attr<bool>("has_y_bool_operand")) {
      x_scalar_operand = static_cast<T>(ctx->Attr<bool>("x_bool_operand"));
      y_scalar_operand = static_cast<T>(ctx->Attr<bool>("y_bool_operand"));
    } else {
      UNIMPLEMENTED() << "The scalar in Where should be bool, float or int";
    }
    WhereKernelUtil<device_type, T, CondT>::WhereXYScalar(
        ctx->stream(), out->shape_view().elem_cnt(), cond->dptr<CondT>(), x_scalar_operand,
        y_scalar_operand, out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WHERE_KERNEL(device_type_v, dtype_pair, ctype_pair)                              \
  REGISTER_USER_KERNEL("where")                                                                   \
      .SetCreateFn<WhereKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                       \
                               OF_PP_PAIR_FIRST(ctype_pair)>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type_v)                                \
                       && (user_op::HobDataType("condition", 0) == OF_PP_PAIR_SECOND(ctype_pair)) \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        Shape* out_shape = ctx->OutputShape("out", 0);                                            \
        const size_t x_bytes =                                                                    \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair)));     \
        const size_t y_bytes =                                                                    \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair)));     \
        const size_t cond_bytes =                                                                 \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(ctype_pair)));     \
        return x_bytes + y_bytes + cond_bytes;                                                    \
      });

#define REGISTER_WHERE_SCALAR_X_KERNEL(device_type_v, dtype_pair, ctype_pair)                     \
  REGISTER_USER_KERNEL("where_scalar_x")                                                          \
      .SetCreateFn<WhereScalarXKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                \
                                      OF_PP_PAIR_FIRST(ctype_pair)>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type_v)                                \
                       && (user_op::HobDataType("condition", 0) == OF_PP_PAIR_SECOND(ctype_pair)) \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        Shape* out_shape = ctx->OutputShape("out", 0);                                            \
        const size_t y_bytes =                                                                    \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair)));     \
        const size_t cond_bytes =                                                                 \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(ctype_pair)));     \
        return y_bytes + cond_bytes;                                                              \
      });

#define REGISTER_WHERE_SCALAR_Y_KERNEL(device_type_v, dtype_pair, ctype_pair)                     \
  REGISTER_USER_KERNEL("where_scalar_y")                                                          \
      .SetCreateFn<WhereScalarYKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                \
                                      OF_PP_PAIR_FIRST(ctype_pair)>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type_v)                                \
                       && (user_op::HobDataType("condition", 0) == OF_PP_PAIR_SECOND(ctype_pair)) \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        Shape* out_shape = ctx->OutputShape("out", 0);                                            \
        const size_t x_bytes =                                                                    \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(dtype_pair)));     \
        const size_t cond_bytes =                                                                 \
            GetCudaAlignedSize(out_shape->elem_cnt() * sizeof(OF_PP_PAIR_FIRST(ctype_pair)));     \
        return x_bytes + cond_bytes;                                                              \
      });

#define REGISTER_WHERE_SCALAR_XY_KERNEL(device_type_v, dtype_pair, ctype_pair)                    \
  REGISTER_USER_KERNEL("where_scalar_xy")                                                         \
      .SetCreateFn<WhereScalarXYKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),               \
                                       OF_PP_PAIR_FIRST(ctype_pair)>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type_v)                                \
                       && (user_op::HobDataType("condition", 0) == OF_PP_PAIR_SECOND(ctype_pair)) \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_SCALAR_X_KERNEL, DEVICE_TYPE_SEQ,
                                 OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)
                                     OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)
                                         BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_SCALAR_Y_KERNEL, DEVICE_TYPE_SEQ,
                                 OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)
                                     OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)
                                         BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_SCALAR_XY_KERNEL, DEVICE_TYPE_SEQ,
                                 OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)
                                     OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)
                                         BOOL_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_KERNEL, (DeviceType::kCUDA), FLOAT16_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
