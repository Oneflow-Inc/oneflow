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

#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

struct NdIndexSliceArgs {
  static const size_t kMaxDims = 8;
  int64_t num_slices;   // The number of slices (indices_shape.Count(0, -1))
  int64_t slice_size;   // The element_cnt of each slice (sliced_shape.Count(indices_num_axes-1))
  int64_t index_ndims;  // The number of dims which are sliced (indices_shape.At(-1))
  int64_t dense_ndims;
  int64_t dense_shape[kMaxDims];
  int64_t dense_stride[kMaxDims];
  int64_t slices_ndims;
  int64_t slices_shape[kMaxDims];
  int64_t slices_stride[kMaxDims];
};

inline NdIndexSliceArgs ConstructNdIndexSliceArgs(const user_op::Tensor& dense,
                                                  const user_op::Tensor& slices,
                                                  const user_op::Tensor& indices) {
  NdIndexSliceArgs args;
  std::memset(&args, 0, sizeof(NdIndexSliceArgs));
  args.num_slices = indices.shape_view().Count(0, indices.shape_view().NumAxes() - 1);
  args.index_ndims = indices.shape_view().At(indices.shape_view().NumAxes() - 1);
  args.slice_size = slices.shape_view().Count(indices.shape_view().NumAxes() - 1);

  args.dense_ndims = dense.shape_view().NumAxes();
  FOR_RANGE(int64_t, i, 0, dense.shape_view().NumAxes()) {
    args.dense_shape[i] = dense.shape_view().At(i);
    args.dense_stride[i] = dense.stride().at(i);
  }
  args.slices_ndims = slices.shape_view().NumAxes();
  FOR_RANGE(int64_t, i, 0, slices.stride().size()) {
    args.slices_shape[i] = slices.shape_view().At(i);
    args.slices_stride[i] = slices.stride().at(i);
  }
  return args;
}

template<DeviceType device_type, typename T, typename I>
struct GatherNdFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* dense, T* slices) const;
};

template<DeviceType device_type, typename T, typename I>
struct ScatterNdAddFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const;
};

template<DeviceType device_type, typename T, typename I>
struct ScatterNdUpdateFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const;
};

template<DeviceType device_type, typename T, typename I>
struct ScatterNdUpdateWithStrideFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const;
};

template<DeviceType device_type, typename T, typename I>
struct FillByNdIndexFunctor final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices, T* dense,
                  T value) const;
};

template<typename I>
OF_DEVICE_FUNC int64_t OffsetInSliceToOffsetInDense(int64_t slice_size, int64_t index_ndims,
                                                    const int64_t* dense_shape, const I* indices,
                                                    int64_t n) {
  int64_t slice_idx = n / slice_size;
  const I* nd_index = indices + slice_idx * index_ndims;
  int64_t offset = 0;
  int64_t product = 1;
  int64_t shifted_index = 0;
  for (int64_t i = index_ndims - 1; i >= 0; --i) {
#if defined(__CUDACC__)
    assert(nd_index[i] < dense_shape[i] && nd_index[i] >= -dense_shape[i] && "index out of bounds");
#else
    CHECK(nd_index[i] < dense_shape[i] && nd_index[i] >= -dense_shape[i])
        << "IndexError: index " << nd_index[i] << " is out of bounds for dimension " << i
        << " with size " << dense_shape[i];
#endif
    shifted_index = nd_index[i] < 0 && nd_index[i] >= -dense_shape[i] ? nd_index[i] + dense_shape[i]
                                                                      : nd_index[i];
    offset += shifted_index * product;
    product *= dense_shape[i];
  }
  return offset * slice_size + n % slice_size;
}

OF_DEVICE_FUNC int64_t GetMemoryOffset4ElementIdx(int64_t n, int64_t ndims, const int64_t* shape,
                                                  const int64_t* stride) {
  int64_t offset = 0;
  for (int64_t i = ndims - 1; i >= 0; --i) {
    offset += n % shape[i] * stride[i];
    n /= shape[i];
  }
  return offset;
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

template<DeviceType device_type, typename T, typename I>
OF_DEVICE_FUNC void DoScatterNdUpdateWithStride(int64_t elem_cnt, const NdIndexSliceArgs& args,
                                                const I* indices, const T* slices, T* dense) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) {
    // dense tensor memory offset
    int64_t dense_index = OffsetInSliceToOffsetInDense(args.slice_size, args.index_ndims,
                                                       args.dense_shape, indices, i);
    int64_t dense_mem_offset = GetMemoryOffset4ElementIdx(dense_index, args.dense_ndims,
                                                          args.dense_shape, args.dense_stride);
    // update tensor memory offset
    int64_t slice_mem_offset =
        GetMemoryOffset4ElementIdx(i, args.slices_ndims, args.slices_shape, args.slices_stride);
    dense[dense_mem_offset] = slices[slice_mem_offset];
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
