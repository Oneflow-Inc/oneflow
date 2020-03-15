#include "oneflow/customized/kernels/where_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<typename T>
T* GetTmpPtr(void* tmp_ptr, size_t offset) {
  CHECK_NOTNULL(tmp_ptr);
  return reinterpret_cast<T*>(static_cast<char*>(tmp_ptr) + offset);
}

template<typename T>
XpuVarNdarray<const T> GetReducedXpuVarNdarray(const user_op::Tensor* reduced_tensor,
                                               const ShapeView& broadcasted_shape) {
  if (reduced_tensor->shape().NumAxes() == broadcasted_shape.NumAxes()) {
    return XpuVarNdarray<const T>(reduced_tensor->shape(), reduced_tensor->dptr<T>());
  } else {
    Shape extended_reduced_shape =
        CreateLeftExtendedShape(reduced_tensor->shape(), broadcasted_shape.NumAxes());
    return XpuVarNdarray<const T>(extended_reduced_shape, reduced_tensor->dptr<T>());
  }
}

template<typename T>
XpuVarNdarray<T> GetMutReducedXpuVarNdarray(user_op::Tensor* reduced_tensor,
                                            const ShapeView& broadcasted_shape) {
  if (reduced_tensor->shape().NumAxes() == broadcasted_shape.NumAxes()) {
    return XpuVarNdarray<T>(reduced_tensor->shape(), reduced_tensor->mut_dptr<T>());
  } else {
    Shape extended_reduced_shape =
        CreateLeftExtendedShape(reduced_tensor->shape(), broadcasted_shape.NumAxes());
    return XpuVarNdarray<T>(extended_reduced_shape, reduced_tensor->mut_dptr<T>());
  }
}

template<DeviceType device_type, typename T>
const T* GetHasBroadcastedPtr(DeviceCtx* ctx, const ShapeView& broadcasted_shape,
                              const user_op::Tensor* reduced_tensor, void* tmp_ptr,
                              size_t* tmp_byte_offset) {
  if (reduced_tensor->shape() == broadcasted_shape) { return reduced_tensor->dptr<T>(); }
  T* cur_tmp_ptr = GetTmpPtr<T>(tmp_ptr, *tmp_byte_offset);
  NdarrayUtil<device_type, T>::BroadcastTo(
      ctx, XpuVarNdarray<T>(broadcasted_shape, cur_tmp_ptr),
      GetReducedXpuVarNdarray<T>(reduced_tensor, broadcasted_shape));
  *tmp_byte_offset += GetCudaAlignedSize(broadcasted_shape.elem_cnt() * sizeof(T));
  return cur_tmp_ptr;
}

template<DeviceType device_type, typename T>
T* GetZeroedBroadcastPtr(DeviceCtx* ctx, const ShapeView& broadcasted_shape,
                         user_op::Tensor* reduced_tensor, void* tmp_ptr, size_t* tmp_byte_offset) {
  T* cur_tmp_ptr = nullptr;
  size_t byte_size = 0;
  if (reduced_tensor->shape() == broadcasted_shape) {
    cur_tmp_ptr = reduced_tensor->mut_dptr<T>();
    byte_size = reduced_tensor->shape().elem_cnt() * GetSizeOfDataType(reduced_tensor->data_type());
  } else {
    cur_tmp_ptr = GetTmpPtr<T>(tmp_ptr, *tmp_byte_offset);
    byte_size = GetCudaAlignedSize(broadcasted_shape.elem_cnt() * sizeof(T));
    *tmp_byte_offset += byte_size;
  }
  Memset<device_type>(ctx, cur_tmp_ptr, 0, byte_size);
  return cur_tmp_ptr;
}

template<DeviceType device_type, typename T>
void TryGetTmpPtrAndDoReduceSum(DeviceCtx* ctx, const ShapeView& broadcasted_shape,
                                const T* broadcasted_ptr, user_op::Tensor* reduced_tensor,
                                void* tmp_ptr, size_t* tmp_byte_offset) {
  if (broadcasted_ptr == reduced_tensor->dptr<T>()) { return; }
  T* cur_tmp_ptr = GetTmpPtr<T>(tmp_ptr, *tmp_byte_offset);
  size_t tmp_byte_size = GetCudaAlignedSize(broadcasted_shape.elem_cnt() * sizeof(T));
  *tmp_byte_offset += tmp_byte_size;
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx, GetMutReducedXpuVarNdarray<T>(reduced_tensor, broadcasted_shape),
      XpuVarNdarray<const T>(broadcasted_shape, broadcasted_ptr),
      XpuVarNdarray<T>(broadcasted_shape, cur_tmp_ptr));
}

size_t GetBroadcastTmpBufferSize(user_op::InferContext* ctx, const std::string& arg_name,
                                 const int32_t arg_index, const Shape* broadcasted_shape) {
  const Shape* shape = ctx->Shape4ArgNameAndIndex(arg_name, arg_index);
  if (*shape == *broadcasted_shape) { return 0; }
  DataType data_type = *ctx->Dtype4ArgNameAndIndex(arg_name, arg_index);
  return GetCudaAlignedSize(broadcasted_shape->elem_cnt() * GetSizeOfDataType(data_type));
}

size_t GetBroadcastAndReduceTmpBufferSize(user_op::InferContext* ctx, const std::string& arg_name,
                                          const int32_t arg_index, const Shape* broadcasted_shape) {
  const Shape* shape = ctx->Shape4ArgNameAndIndex(arg_name, arg_index);
  if (*shape == *broadcasted_shape) { return 0; }
  DataType data_type = *ctx->Dtype4ArgNameAndIndex(arg_name, arg_index);
  size_t broadcast_tmp_buffer_size =
      GetCudaAlignedSize(broadcasted_shape->elem_cnt() * GetSizeOfDataType(data_type));
  size_t reduce_tmp_buffer_size =
      GetCudaAlignedSize(broadcasted_shape->elem_cnt() * GetSizeOfDataType(data_type));
  return broadcast_tmp_buffer_size + reduce_tmp_buffer_size;
}

}  // namespace

template<DeviceType device_type, typename T, typename CondT>
void WhereKernel<device_type, T, CondT>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
  size_t tmp_byte_offset = 0;
  const CondT* cond_ptr = GetHasBroadcastedPtr<device_type, CondT>(ctx->device_ctx(), out->shape(),
                                                                   cond, tmp_ptr, &tmp_byte_offset);
  const T* x_ptr = GetHasBroadcastedPtr<device_type, T>(ctx->device_ctx(), out->shape(), x, tmp_ptr,
                                                        &tmp_byte_offset);
  const T* y_ptr = GetHasBroadcastedPtr<device_type, T>(ctx->device_ctx(), out->shape(), y, tmp_ptr,
                                                        &tmp_byte_offset);
  if (tmp) {
    CHECK_LE(tmp_byte_offset, tmp->shape().elem_cnt() * GetSizeOfDataType(tmp->data_type()));
  } else {
    CHECK_EQ(tmp_byte_offset, 0);
  }
  WhereFunctor<device_type, T, CondT>()(ctx->device_ctx(), out->shape().elem_cnt(), cond_ptr, x_ptr,
                                        y_ptr, out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename CondT>
void WhereGradKernel<device_type, T, CondT>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
  const user_op::Tensor* dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
  user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
  user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  void* tmp_ptr = tmp ? tmp->mut_dptr() : nullptr;
  size_t tmp_byte_offset = 0;
  const CondT* broadcasted_cond_ptr = GetHasBroadcastedPtr<device_type, CondT>(
      ctx->device_ctx(), dz->shape(), cond, tmp_ptr, &tmp_byte_offset);
  T* broadcasted_dx_ptr = GetZeroedBroadcastPtr<device_type, T>(ctx->device_ctx(), dz->shape(), dx,
                                                                tmp_ptr, &tmp_byte_offset);
  T* broadcasted_dy_ptr = GetZeroedBroadcastPtr<device_type, T>(ctx->device_ctx(), dz->shape(), dy,
                                                                tmp_ptr, &tmp_byte_offset);
  WhereGradFunctor<device_type, T, CondT>()(ctx->device_ctx(), dz->shape().elem_cnt(),
                                            broadcasted_cond_ptr, dz->dptr<T>(), broadcasted_dx_ptr,
                                            broadcasted_dy_ptr);
  TryGetTmpPtrAndDoReduceSum<device_type>(ctx->device_ctx(), dz->shape(), broadcasted_dx_ptr, dx,
                                          tmp_ptr, &tmp_byte_offset);
  TryGetTmpPtrAndDoReduceSum<device_type>(ctx->device_ctx(), dz->shape(), broadcasted_dy_ptr, dy,
                                          tmp_ptr, &tmp_byte_offset);
  if (tmp) {
    CHECK_LE(tmp_byte_offset, tmp->shape().elem_cnt() * GetSizeOfDataType(tmp->data_type()));
  } else {
    CHECK_EQ(tmp_byte_offset, 0);
  }
}

size_t InferWhereTmpBufferSize(user_op::InferContext* ctx) {
  const Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
  size_t tmp_buffer_size = 0;
  tmp_buffer_size += GetBroadcastTmpBufferSize(ctx, "condition", 0, out_shape);
  tmp_buffer_size += GetBroadcastTmpBufferSize(ctx, "x", 0, out_shape);
  tmp_buffer_size += GetBroadcastTmpBufferSize(ctx, "y", 0, out_shape);
  return tmp_buffer_size;
}

size_t InferWhereGradTmpBufferSize(user_op::InferContext* ctx) {
  const Shape* dz_shape = ctx->Shape4ArgNameAndIndex("dz", 0);
  size_t tmp_buffer_size = 0;
  tmp_buffer_size += GetBroadcastAndReduceTmpBufferSize(ctx, "condition", 0, dz_shape);
  tmp_buffer_size += GetBroadcastAndReduceTmpBufferSize(ctx, "dx", 0, dz_shape);
  tmp_buffer_size += GetBroadcastAndReduceTmpBufferSize(ctx, "dy", 0, dz_shape);
  return tmp_buffer_size;
}

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kCPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    DoWhere(elem_cnt, cond, lhs, rhs, out);
  }
};

template<typename T, typename CondT>
struct WhereGradFunctor<DeviceType::kCPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* grad,
                  T* lhs_grad, T* rhs_grad) const {
    DoWhereGrad(elem_cnt, cond, grad, lhs_grad, rhs_grad);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTORS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_KERNELS, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

}  // namespace oneflow
