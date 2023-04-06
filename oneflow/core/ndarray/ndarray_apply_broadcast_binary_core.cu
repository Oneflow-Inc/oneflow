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
#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"

namespace oneflow {

namespace {

template<typename Index>
struct XY2XFunctor final {
  __host__ __device__ XY2XFunctor(Index dim_y) : dim_y_(dim_y) {}

  __host__ __device__ Index operator()(Index idx) const { return idx / dim_y_; }

  Index dim_y_;
};

template<typename Index>
struct XY2YFunctor final {
  __host__ __device__ XY2YFunctor(Index dim_y) : dim_y_(dim_y) {}

  __host__ __device__ Index operator()(Index idx) const { return idx % dim_y_; }

  Index dim_y_;
};

template<typename Index>
struct XYZ2XZFunctor final {
  __host__ __device__ XYZ2XZFunctor(Index dim_y, Index dim_z)
      : dim_yz_(dim_y * dim_z), dim_z_(dim_z) {}

  __host__ __device__ Index operator()(Index idx) const {
    const Index x = idx / dim_yz_;
    const Index z = (idx % dim_yz_) % dim_z_;
    return x * dim_z_ + z;
  }

  Index dim_yz_;
  Index dim_z_;
};

template<typename Index>
struct XYZ2YFunctor final {
  __host__ __device__ XYZ2YFunctor(Index dim_y, Index dim_z)
      : dim_yz_(dim_y * dim_z), dim_z_(dim_z) {}

  __host__ __device__ Index operator()(Index idx) const { return (idx % dim_yz_) / dim_z_; }

  Index dim_yz_;
  Index dim_z_;
};

template<typename T, typename K, template<typename> class binary_func, typename OffsetFunctor>
__global__ void PartialBroadcastGpu(K n, typename BinaryFuncTrait<binary_func, T>::return_type* y,
                                    const T* a, const T* b, OffsetFunctor offset_functor) {
  CUDA_1D_KERNEL_LOOP_T(K, i, n) { y[i] = binary_func<T>::Invoke(a[i], b[offset_functor(i)]); }
}

template<typename T, int NDIMS, template<typename> class binary_func>
__global__ void GpuBroadcastBinaryFunc(
    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type> y,
    const XpuVarNdarray<const T> a, const XpuVarNdarray<const T> b) {
  NdarrayApplyBroadcastBinaryCore<T, NDIMS, binary_func>::Apply(y, a, b);
}
template<typename T, int NDIMS, template<typename> class binary_func>
__global__ void GpuInplaceBroadcastBinaryFunc(const XpuVarNdarray<T> y,
                                              const XpuVarNdarray<const T> x) {
  NdarrayApplyBroadcastBinaryCore<T, NDIMS, binary_func>::InplaceApply(y, x);
}

}  // namespace

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinaryCoreWrapper<DeviceType::kCUDA, T, NDIMS, binary_func> final {
  static void Apply(ep::Stream* stream,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    size_t n = y.host_shape().HostElemNum();
    if (n == 0) { return; }
    if (IsKernelSafeInt32(n) && PartialBroadcast<int32_t>(stream, y, a, b)) { return; }
    if (!IsKernelSafeInt32(n) && PartialBroadcast<int64_t>(stream, y, a, b)) { return; }
    RUN_CUDA_KERNEL((GpuBroadcastBinaryFunc<T, NDIMS, binary_func>), stream, n, y, a, b);
  }

  template<typename K>
  static bool PartialBroadcast(
      ep::Stream* stream,
      const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    size_t n = y.host_shape().HostElemNum();
    if (y.host_shape() == a.host_shape()) {
      if (y.host_shape().NumAxes() == 2) {
        const K y_dim0 = y.host_shape().At(0);
        const K y_dim1 = y.host_shape().At(1);
        const K b_dim0 = b.host_shape().At(0);
        const K b_dim1 = b.host_shape().At(1);
        if (b_dim0 == y_dim0 && b_dim1 == 1) {
          XY2XFunctor<K> xy2x(y_dim1);
          RUN_CUDA_KERNEL((PartialBroadcastGpu<T, K, binary_func, XY2XFunctor<K>>), stream, n, n,
                          y.host_ptr(), a.host_ptr(), b.host_ptr(), xy2x);
          return true;
        }
        if (b_dim0 == 1 && b_dim1 == y_dim1) {
          XY2YFunctor<K> xy2y(y_dim1);
          RUN_CUDA_KERNEL((PartialBroadcastGpu<T, K, binary_func, XY2YFunctor<K>>), stream, n, n,
                          y.host_ptr(), a.host_ptr(), b.host_ptr(), xy2y);
          return true;
        }
      }
      if (y.host_shape().NumAxes() == 3) {
        const K y_dim0 = y.host_shape().At(0);
        const K y_dim1 = y.host_shape().At(1);
        const K y_dim2 = y.host_shape().At(2);
        const K b_dim0 = b.host_shape().At(0);
        const K b_dim1 = b.host_shape().At(1);
        const K b_dim2 = b.host_shape().At(2);
        if (b_dim0 == y_dim0 && b_dim1 == 1 && b_dim2 == y_dim2) {
          XYZ2XZFunctor<K> xyz2xz(y_dim1, y_dim2);
          RUN_CUDA_KERNEL((PartialBroadcastGpu<T, K, binary_func, XYZ2XZFunctor<K>>), stream, n, n,
                          y.host_ptr(), a.host_ptr(), b.host_ptr(), xyz2xz);
          return true;
        }
        if (b_dim0 == 1 && b_dim1 == y_dim1 && b_dim2 == 1) {
          XYZ2YFunctor<K> xyz2y(y_dim1, y_dim2);
          RUN_CUDA_KERNEL((PartialBroadcastGpu<T, K, binary_func, XYZ2YFunctor<K>>), stream, n, n,
                          y.host_ptr(), a.host_ptr(), b.host_ptr(), xyz2y);
          return true;
        }
      }
    }
    return false;
  }
};

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastInplaceBinaryCoreWrapper<DeviceType::kCUDA, T, NDIMS, binary_func>
    final {
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    size_t n = y.host_shape().HostElemNum();
    XpuVarNdarray<const T> a(y.host_shape(), y.host_ptr());
    using NBB = NdarrayApplyBroadcastBinaryCoreWrapper<DeviceType::kCUDA, T, NDIMS, binary_func>;
    if (n == 0) { return; }
    if (IsKernelSafeInt32(n) && NBB::template PartialBroadcast<int32_t>(stream, y, a, x)) {
      return;
    }
    if (!IsKernelSafeInt32(n) && NBB::template PartialBroadcast<int64_t>(stream, y, a, x)) {
      return;
    }
    RUN_CUDA_KERNEL((GpuInplaceBroadcastBinaryFunc<T, NDIMS, binary_func>), stream, n, y, x);
  }
};

}  // namespace oneflow
