#include "oneflow/customized/kernels/where_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
inline const T* GetBroadcastedPtr(DeviceCtx* ctx, const user_op::Tensor* reduced_tensor,
                                  const ShapeView& broadcasted_shape, void* tmp_ptr,
                                  size_t* tmp_byte_offset) {
  if (reduced_tensor->shape() == broadcasted_shape) { return reduced_tensor->dptr<T>(); }
  T* cur_tmp_ptr = reinterprete_cast<T*>(tmp_ptr + *tmp_byte_offset);
  NdarrayUtil<device_type, T>::BroadcastTo(
      ctx, XpuVarNdarray<T>(broadcasted_shape, cur_tmp_ptr),
      XpuVarNdarray<const T>(reduced_tensor->shape(), reduced_tensor->dptr<T>()));
  *tmp_byte_offset += GetCudaAlignedSize(broadcasted_shape.elem_cnt() * sizeof(T));
  return cur_tmp_ptr;
}

}  // namespace

template<DeviceType device_type, typename T, typename CondT>
void WhereKernel<device_type, T, CondT>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
  size_t tmp_byte_offset = 0;
  const CondT* cond_ptr = GetBroadcastedPtr<device_type, CondT>(
      ctx->device_ctx(), cond, out->shape(), tmp->mut_dptr(), &tmp_byte_offset);
  const T* x_ptr = GetBroadcastedPtr<device_type, T>(ctx->device_ctx(), x, out->shape(),
                                                     tmp->mut_dptr(), &tmp_byte_offset);
  const T* y_ptr = GetBroadcastedPtr<device_type, T>(ctx->device_ctx(), y, out->shape(),
                                                     tmp->mut_dptr(), &tmp_byte_offset);
  CHECK_LE(tmp_byte_offset, tmp->shape().elem_cnt() * GetSizeOfDataType(tmp->data_type()));
  WhereFunctor<device_type, T, CondT>()(ctx->device_ctx(), out->shape().elem_cnt(), cond_ptr, x_ptr,
                                        y_ptr, out->mut_dptr<T>());
}

size_t InferWhereTmpBufferSize(oneflow::user_op::InferContext* ctx) {
  const Shape* cond_shape = ctx->Shape4ArgNameAndIndex("condition", 0);
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
  const Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
  size_t tmp_buffer_size = 0;
  if (*cond_shape != *out_shape) {
    tmp_buffer_size += GetCudaAlignedSize(
        out_shape->elem_cnt() * GetSizeOfDataType(*ctx->Dtype4ArgNameAndIndex("condition", 0)));
  }
  if (*x_shape != *out_shape) {
    tmp_buffer_size += GetCudaAlignedSize(out_shape->elem_cnt()
                                          * GetSizeOfDataType(*ctx->Dtype4ArgNameAndIndex("x", 0)));
  }
  if (*y_shape != *out_shape) {
    tmp_buffer_size += GetCudaAlignedSize(out_shape->elem_cnt()
                                          * GetSizeOfDataType(*ctx->Dtype4ArgNameAndIndex("y", 0)));
  }
  return tmp_buffer_size;
}

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kGPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    DoWhere(elem_cnt, cond, lhs, rhs, out);
  }
};

}  // namespace oneflow
