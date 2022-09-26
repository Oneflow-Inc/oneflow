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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/ndarray/xpu_ndarray_assign.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

#define DECLARE_NDARRAY_REDUCE_IMPL(struct_name)                                       \
  template<DeviceType device_type, typename T, template<typename> class binary_func>   \
  struct struct_name final {                                                           \
    static bool Matched(                                                               \
        const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y, \
        const XpuVarNdarray<const T>& x);                                              \
    static void Reduce(                                                                \
        ep::Stream* ctx,                                                               \
        const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y, \
        const XpuVarNdarray<const T>& x, const XpuVarNdarray<T>& tmp_storage);         \
  }
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayScalarReduce);
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayMatrixRowReduce);
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayMatrixColReduce);
DECLARE_NDARRAY_REDUCE_IMPL(NdarrayXYZCubeXZReduce);
#undef DECLARE_NDARRAY_REDUCE_IMPL

template<DeviceType device_type, typename T, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayNoReduce;

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayNoReduce<device_type, T, binary_func,
                       typename std::enable_if<std::is_same<
                           T, typename BinaryFuncTrait<binary_func, T>::return_type>::value>::type>
    final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {
    return x.shape() == y.shape();
  }
  static void Reduce(ep::Stream* ctx, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    if (std::is_same<binary_func<T>, BinaryFuncNanSum<T>>()) {
      XpuNdarrayAssign<device_type, RetT>::AssignNanSum(ctx, y, x);
    } else {
      XpuNdarrayAssign<device_type, RetT>::Assign(ctx, y, x);
    }
  }
};

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayNoReduce<device_type, T, binary_func,
                       typename std::enable_if<!std::is_same<
                           T, typename BinaryFuncTrait<binary_func, T>::return_type>::value>::type>
    final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {
    return x.shape() == y.shape();
  }
  static void Reduce(ep::Stream* ctx, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    return SwitchReduce(SwitchCase(y.shape().NumAxes()), ctx, y, x, tmp_storage);
  }

 private:
#define DEFINE_NDARRAY_REDUCE(func_name, NDIMS) func_name<NDIMS>
  DEFINE_STATIC_SWITCH_FUNC(void, Reduce, DEFINE_NDARRAY_REDUCE, MAKE_NDIM_CTRV_SEQ(DIM_SEQ))
#undef DEFINE_NDARRAY_REDUCE

  template<int NDIMS>
  static void Reduce(ep::Stream* ctx, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    XpuNdarrayAssign<device_type, RetT>::template Assign<NDIMS>(ctx, y, x);
  }
};

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayReduceCoreWrapper final {
  static void ReduceAxis(ep::Stream* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis);
};

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayDefaultReduce final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static void Reduce(ep::Stream* ctx, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    return SwitchReduce(SwitchCase(y.shape().NumAxes()), ctx, y, x, tmp_storage);
  }

 private:
#define DEFINE_NDARRAY_REDUCE(func_name, NDIMS) func_name<NDIMS>
  DEFINE_STATIC_SWITCH_FUNC(void, Reduce, DEFINE_NDARRAY_REDUCE, MAKE_NDIM_CTRV_SEQ(DIM_SEQ))
#undef DEFINE_NDARRAY_REDUCE

  template<int NDIMS>
  static void Reduce(ep::Stream* ctx, const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    XpuVarNdarray<T> storage(x.shape(), tmp_storage.ptr());
    XpuShape cur_shape(x.shape());
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    CHECK(x.shape() != y.shape());
    XpuNdarrayAssign<device_type, T>::Assign(ctx, storage, x);
    for (int i = 0; i < x.shape().NumAxes(); ++i) {
      if (y.shape().At(i) == x.shape().At(i)) { continue; }
      CHECK_EQ(y.shape().At(i), 1);
      CHECK_GT(x.shape().At(i), y.shape().At(i));
      InplaceReduceAxis<NDIMS>(ctx, i, storage, &cur_shape);
    }
    XpuReducedNdarray<T, NDIMS> reduced(y.shape(), storage);
    XpuNdarrayAssign<device_type, RetT>::template Assign<NDIMS>(ctx, y, reduced);
  }

  template<int NDIMS>
  static void InplaceReduceAxis(ep::Stream* ctx, int axis, const XpuVarNdarray<T>& implace,
                                XpuShape* cur_shape) {
    int64_t target_elem_num = cur_shape->ElemNum() / cur_shape->At(axis);
    while (cur_shape->At(axis) > 1) {
      int64_t shrink = 8 + std::sqrt(target_elem_num);
      XpuReducedNdarray<T, NDIMS> from(*cur_shape, implace);
      int64_t new_dim_value = (cur_shape->At(axis) + (shrink - 1)) / shrink;
      cur_shape->Set(axis, new_dim_value);
      XpuReducedNdarray<T, NDIMS> to(*cur_shape, implace);
      NdarrayReduceCoreWrapper<device_type, T, NDIMS, binary_func>::ReduceAxis(ctx, to, from, axis);
    }
  }
};

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayReduceCore final {
  template<typename X>
  OF_DEVICE_FUNC static void ReduceAxis(const XpuReducedNdarray<T, NDIMS>& dst_reduced, const X& x,
                                        int axis) {
    size_t n = dst_reduced.shape().ElemNum();
    int64_t dst_dim_val = dst_reduced.shape().At(axis);
    XPU_1D_KERNEL_LOOP_BEGIN(i, n);
    T* dst_reduced_ptr = dst_reduced.template Mut(i);
    int64_t coord[NDIMS];
    dst_reduced.shape().template Offset2Coordinate<NDIMS>(i, coord);
    T reduced = UnitOfBinaryFunc<T, binary_func>::Val();
    while (coord[axis] < x.shape().At(axis)) {
      reduced = binary_func<T>::Invoke(reduced, x.template Get<NDIMS>(coord));
      coord[axis] += dst_dim_val;
    }
    *dst_reduced_ptr = reduced;
    XPU_1D_KERNEL_LOOP_END();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
