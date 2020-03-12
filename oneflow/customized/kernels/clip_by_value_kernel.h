#ifndef ONEFLOW_CUSTOMIZED_KERNELS_CLIP_BY_VALUE_KERNEL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_CLIP_BY_VALUE_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct ClipValuesUtil {
  static void ByMin(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                    T* out_ptr);
  static void ByMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* max_value,
                    T* out_ptr);
  static void ByMinMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                       const T* max_value, T* out_ptr);
};

template<DeviceType device_type, typename T>
struct ClipGradUtil {
  static void ByMin(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                    T* grad_ptr);
  static void ByMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* max_value,
                    T* grad_ptr);
  static void ByMinMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                       const T* max_value, T* grad_ptr);
};

template<DeviceType device_type, typename T>
struct DeviceClip {
  OF_DEVICE_FUNC static T Min(const T value, const T min_value);
  OF_DEVICE_FUNC static T Max(const T value, const T max_value);
};

template<DeviceType device_type, typename T>
OF_DEVICE_FUNC void ClipValuesByMinMax(const int64_t num_values, const T* values, const T min_value,
                                       const T max_value, T* out_ptr) {
  XPU_1D_KERNEL_LOOP(i, num_values) {
    out_ptr[i] = DeviceClip<device_type, T>::Min(
        DeviceClip<device_type, T>::Max(values[i], min_value), max_value);
  }
}

template<DeviceType device_type, typename T>
OF_DEVICE_FUNC void ClipValuesByMin(const int64_t num_values, const T* values, const T min_value,
                                    T* out_ptr) {
  XPU_1D_KERNEL_LOOP(i, num_values) {
    out_ptr[i] = DeviceClip<device_type, T>::Max(values[i], min_value);
  }
}

template<DeviceType device_type, typename T>
OF_DEVICE_FUNC void ClipValuesByMax(const int64_t num_values, const T* values, const T max_value,
                                    T* out_ptr) {
  XPU_1D_KERNEL_LOOP(i, num_values) {
    out_ptr[i] = DeviceClip<device_type, T>::Min(values[i], max_value);
  }
}

template<DeviceType device_type, typename T>
OF_DEVICE_FUNC void ClipGradByMinMax(const int64_t num_values, const T* values, const T min_value,
                                     const T max_value, T* grad_ptr) {
  XPU_1D_KERNEL_LOOP(i, num_values) {
    if (values[i] < min_value || values[i] > max_value) { grad_ptr[i] = GetZeroVal<T>(); }
  }
}

template<DeviceType device_type, typename T>
OF_DEVICE_FUNC void ClipGradByMin(const int64_t num_values, const T* values, const T min_value,
                                  T* grad_ptr) {
  XPU_1D_KERNEL_LOOP(i, num_values) {
    if (values[i] < min_value) { grad_ptr[i] = GetZeroVal<T>(); }
  }
}

template<DeviceType device_type, typename T>
OF_DEVICE_FUNC void ClipGradByMax(const int64_t num_values, const T* values, const T max_value,
                                  T* grad_ptr) {
  XPU_1D_KERNEL_LOOP(i, num_values) {
    if (values[i] > max_value) { grad_ptr[i] = GetZeroVal<T>(); }
  }
}

template<DeviceType device_type, typename T>
class ClipByValueKernel final : public user_op::OpKernel {
 public:
  ClipByValueKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ClipByValueKernel() = default;
  ~ClipByValueKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T>
class ClipByValueGradKernel final : public user_op::OpKernel {
 public:
  ClipByValueGradKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ClipByValueGradKernel() = default;
  ~ClipByValueGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T>
void ClipByValueKernel<device_type, T>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  const user_op::Tensor* min = ctx->Tensor4ArgNameAndIndex("min", 0);
  const user_op::Tensor* max = ctx->Tensor4ArgNameAndIndex("max", 0);
  user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

  if (x->dptr<T>() != y->mut_dptr<T>()) {
    size_t out_bytes_size = y->shape().elem_cnt() * GetSizeOfDataType(y->data_type());
    Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>(), x->dptr<T>(), out_bytes_size);
  }

  if (min != nullptr && max != nullptr) {
    ClipValuesUtil<device_type, T>::ByMinMax(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                             min->dptr<T>(), max->dptr<T>(), y->mut_dptr<T>());
  } else if (min != nullptr) {
    ClipValuesUtil<device_type, T>::ByMin(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                          min->dptr<T>(), y->mut_dptr<T>());
  } else if (max != nullptr) {
    ClipValuesUtil<device_type, T>::ByMax(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                          max->dptr<T>(), y->mut_dptr<T>());
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void ClipByValueGradKernel<device_type, T>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  const user_op::Tensor* min = ctx->Tensor4ArgNameAndIndex("min", 0);
  const user_op::Tensor* max = ctx->Tensor4ArgNameAndIndex("max", 0);
  user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

  if (dy->dptr<T>() != dx->mut_dptr<T>()) {
    size_t dx_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memcpy<device_type>(ctx->device_ctx(), dx->mut_dptr<T>(), dy->dptr<T>(), dx_bytes_size);
  }

  if (min != nullptr && max != nullptr) {
    ClipGradUtil<device_type, T>::ByMinMax(ctx->device_ctx(), dx->shape().elem_cnt(), x->dptr<T>(),
                                           min->dptr<T>(), max->dptr<T>(), dx->mut_dptr<T>());
  } else if (min != nullptr) {
    ClipGradUtil<device_type, T>::ByMin(ctx->device_ctx(), dx->shape().elem_cnt(), x->dptr<T>(),
                                        min->dptr<T>(), dx->mut_dptr<T>());
  } else if (max != nullptr) {
    ClipGradUtil<device_type, T>::ByMax(ctx->device_ctx(), dx->shape().elem_cnt(), x->dptr<T>(),
                                        max->dptr<T>(), dx->mut_dptr<T>());
  } else {
    UNIMPLEMENTED();
  }
}

#define REGISTER_CLIP_KERNEL(op_type_name, kernel, input_name, output_name, device_type_v, dtype) \
  REGISTER_USER_KERNEL(#op_type_name)                                                             \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                           \
        return new kernel<device_type_v, dtype>(ctx);                                             \
      })                                                                                          \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                       \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex(#output_name, 0);    \
        if (ctx.device_type() == device_type_v                                                    \
            && out_desc->data_type() == GetDataType<dtype>::value) {                              \
          return true;                                                                            \
        }                                                                                         \
        return false;                                                                             \
      })                                                                                          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                      \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {   \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn(#output_name, 0, #input_name, 0, true));           \
        return Maybe<void>::Ok();                                                                 \
      });

#define REGISTER_CLIP_KERNELS(device_type_v, dtype_pair)                                 \
  REGISTER_CLIP_KERNEL(clip_by_value, ClipByValueKernel, x, y, device_type_v,            \
                       OF_PP_PAIR_FIRST(dtype_pair))                                     \
  REGISTER_CLIP_KERNEL(clip_by_value_grad, ClipByValueGradKernel, dy, dx, device_type_v, \
                       OF_PP_PAIR_FIRST(dtype_pair))

#define INSTANTIATE_CLIP_UTIL(device_type_v, dtype_pair)                       \
  template struct DeviceClip<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;     \
  template struct ClipValuesUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>; \
  template struct ClipGradUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_CLIP_BY_VALUE_KERNEL_H_
