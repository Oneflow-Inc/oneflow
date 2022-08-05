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
#ifndef ONEFLOW_USER_KERNELS_ND_INDEX_SLICE_UTIL_H_
#define ONEFLOW_USER_KERNELS_ND_INDEX_SLICE_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T, typename I>
struct NdIndexSliceArgs {
  static const size_t kMaxDims = 8;
  int64_t num_slices;
  int64_t slice_size;
  int64_t index_ndims;
  int64_t dense_shape[kMaxDims];
};

template<typename T, typename I>
inline NdIndexSliceArgs<T, I> ConstructNdIndexSliceArgs(const user_op::Tensor& dense,
                                                        const user_op::Tensor& slices,
                                                        const user_op::Tensor& indices) {
  NdIndexSliceArgs<T, I> args;
  std::memset(&args, 0, sizeof(NdIndexSliceArgs<T, I>));
  args.num_slices = indices.shape_view().Count(0, indices.shape_view().NumAxes() - 1);
  args.index_ndims = indices.shape_view().At(indices.shape_view().NumAxes() - 1);
  args.slice_size = slices.shape_view().Count(indices.shape_view().NumAxes() - 1);
  FOR_RANGE(int64_t, i, 0, dense.shape_view().NumAxes()) {
    args.dense_shape[i] = dense.shape_view().At(i);
  }
  return args;
}

template<DeviceType device_type, typename T, typename I>
struct GatherNdFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* dense, T* slices) const;
};

template<DeviceType device_type, typename T, typename I>
struct ScatterNdAddFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* slices, T* dense) const;
};

template<DeviceType device_type, typename T, typename I>
struct ScatterNdUpdateFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* slices, T* dense) const;
};

template<DeviceType device_type, typename T, typename I>
struct FillByNdIndexFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  T* dense, T value) const;
};

template<typename I>
OF_DEVICE_FUNC int64_t OffsetInSliceToOffsetInDense(int64_t slice_size, int64_t index_ndims,
                                                    const int64_t* dense_shape, const I* indices,
                                                    int64_t n) {
  int64_t slice_idx = n / slice_size;
  const I* cur_nd_index_ptr = indices + slice_idx * index_ndims;
  int64_t offset = 0;
  int64_t product = 1;
  for (int64_t i = index_ndims - 1; i >= 0; --i) {
    offset += cur_nd_index_ptr[i] * product;
    product *= dense_shape[i];
  }
  return offset * slice_size + n % slice_size;
}

template<typename T, typename I>
OF_DEVICE_FUNC void DoGatherNd(int64_t elem_cnt, int64_t slice_size, int64_t index_ndims,
                               const int64_t* dense_shape, const I* indices, const T* dense,
                               T* slices) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = OffsetInSliceToOffsetInDense(slice_size, index_ndims, dense_shape, indices, i);
    slices[i] = dense[offset];
  }
}

template<DeviceType device_type, typename T>
struct DeviceAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { *y += *x; }
};

template<DeviceType device_type, typename T, typename I>
OF_DEVICE_FUNC void DoScatterNdAdd(int64_t elem_cnt, int64_t slice_size, int64_t index_ndims,
                                   const int64_t* dense_shape, const I* indices, const T* slices,
                                   T* dense) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = OffsetInSliceToOffsetInDense(slice_size, index_ndims, dense_shape, indices, i);
    DeviceAdd<device_type, T>::Invoke(slices + i, dense + offset);
  }
}

template<DeviceType device_type, typename T, typename I>
OF_DEVICE_FUNC void DoScatterNdUpdate(int64_t elem_cnt, int64_t slice_size, int64_t index_ndims,
                                      const int64_t* dense_shape, const I* indices, const T* slices,
                                      T* dense) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = OffsetInSliceToOffsetInDense(slice_size, index_ndims, dense_shape, indices, i);
    dense[offset] = slices[i];
  }
}

template<typename T, typename I>
OF_DEVICE_FUNC void DoFillByNdIndex(int64_t elem_cnt, int64_t slice_size, int64_t index_ndims,
                                    const int64_t* dense_shape, const I* indices, T* dense,
                                    T value) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t offset = OffsetInSliceToOffsetInDense(slice_size, index_ndims, dense_shape, indices, i);
    dense[offset] = value;
  }
}

#define INSTANTIATE_GATHER_ND_FUNCTOR(device_type_v, dtype_pair, itype_pair)   \
  template struct GatherNdFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                  OF_PP_PAIR_FIRST(itype_pair)>;

#define INSTANTIATE_SCATTER_ND_ADD_FUNCTOR(device_type_v, dtype_pair, itype_pair)  \
  template struct ScatterNdAddFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                      OF_PP_PAIR_FIRST(itype_pair)>;

#define INSTANTIATE_FILL_BY_ND_INDEX_FUNCTOR(device_type_v, dtype_pair, itype_pair) \
  template struct FillByNdIndexFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                       OF_PP_PAIR_FIRST(itype_pair)>;

#define INSTANTIATE_ND_INDEX_SLICE_FUNCTORS(device_type_v, dtype_pair, itype_pair) \
  INSTANTIATE_GATHER_ND_FUNCTOR(device_type_v, dtype_pair, itype_pair)             \
  INSTANTIATE_SCATTER_ND_ADD_FUNCTOR(device_type_v, dtype_pair, itype_pair)        \
  INSTANTIATE_FILL_BY_ND_INDEX_FUNCTOR(device_type_v, dtype_pair, itype_pair)

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_ND_INDEX_SLICE_UTIL_H_
