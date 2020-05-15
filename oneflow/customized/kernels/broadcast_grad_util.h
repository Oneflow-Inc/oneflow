#ifndef ONEFLOW_CUSTOMIZED_KERNELS_BROADCAST_GRAD_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_BROADCAST_GRAD_UTIL_H_
#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

namespace user_op {

template<typename T, DeviceType device_type>
struct BroadcastDivGrad {
  static void XGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dx, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::BroadcastDiv(
        ctx, tmp, dz, XpuVarNdarray<const T>(tensor_y->shape(), tensor_y->dptr<T>(), num_axes));
    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dx->shape(), tensor_dx->mut_dptr<T>(), num_axes), const_tmp,
        tmp);
  }

  static void YGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dy, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::BroadcastDiv(
        ctx, tmp, XpuVarNdarray<const T>(tensor_x->shape(), tensor_x->dptr<T>(), num_axes),
        XpuVarNdarray<const T>(tensor_y->shape(), tensor_y->dptr<T>(), num_axes));
    NdarrayUtil<device_type, T>::InplaceBroadcastDiv(
        ctx, tmp, XpuVarNdarray<const T>(tensor_y->shape(), tensor_y->dptr<T>(), num_axes));
    NdarrayUtil<device_type, T>::BroadcastMul(ctx, tmp, dz, const_tmp);
    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dy->shape(), tensor_dy->mut_dptr<T>(), num_axes), const_tmp,
        tmp);
    NdarrayUtil<device_type, T>::InplaceNegative(
        ctx, XpuVarNdarray<T>(tensor_dy->shape(), tensor_dy->mut_dptr<T>(), num_axes));
  }
};

template<typename T, DeviceType device_type>
struct BroadcastAddGrad {
  static void XGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dx, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dx->shape(), tensor_dx->mut_dptr<T>(), num_axes), dz, tmp);
  }

  static void YGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dy, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dy->shape(), tensor_dy->mut_dptr<T>(), num_axes), dz, tmp);
  }
};

template<typename T, DeviceType device_type>
struct BroadcastSubGrad {
  static void XGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dx, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dx->shape(), tensor_dx->mut_dptr<T>(), num_axes), dz, tmp);
  }

  static void YGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dy, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dy->shape(), tensor_dy->mut_dptr<T>(), num_axes), dz, tmp);
    NdarrayUtil<device_type, T>::InplaceNegative(
        ctx, XpuVarNdarray<T>(tensor_dy->shape(), tensor_dy->mut_dptr<T>(), num_axes));
  }
};

template<typename T, DeviceType device_type>
struct BroadcastMulGrad {
  static void XGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dx, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::BroadcastMul(
        ctx, tmp, dz, XpuVarNdarray<const T>(tensor_y->shape(), tensor_y->dptr<T>(), num_axes));
    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dx->shape(), tensor_dx->mut_dptr<T>(), num_axes), const_tmp,
        tmp);
  }

  static void YGrad(DeviceCtx* ctx, const Tensor* tensor_dz, Tensor* tensor_dy, Tensor* tmp_buffer,
                    const Tensor* tensor_x, const Tensor* tensor_y) {
    size_t num_axes = tensor_dz->shape().NumAxes();
    XpuVarNdarray<const T> dz(tensor_dz->shape(), tensor_dz->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device_type, T>::BroadcastMul(
        ctx, tmp, dz, XpuVarNdarray<const T>(tensor_x->shape(), tensor_y->dptr<T>(), num_axes));
    NdarrayUtil<device_type, T>::ReduceSum(
        ctx, XpuVarNdarray<T>(tensor_dy->shape(), tensor_dy->mut_dptr<T>(), num_axes), const_tmp,
        tmp);
  }
};

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_BROADCAST_GRAD_UTIL_H_