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
#include "oneflow/user/kernels/nd_index_slice_kernels.h"

namespace oneflow {

template<typename T, typename I>
struct GatherNdFunctor<DeviceType::kCPU, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* dense, T* slices) const {
    DoGatherNd(args.num_slices * args.slice_size, args.slice_size, args.index_ndims,
               args.dense_shape, indices, dense, slices);
  }
};

template<typename T, typename I>
struct ScatterNdAddFunctor<DeviceType::kCPU, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const {
    DoScatterNdAdd<DeviceType::kCPU>(args.num_slices * args.slice_size, args.slice_size,
                                     args.index_ndims, args.dense_shape, indices, slices, dense);
  }
};

template<typename T, typename I>
struct ScatterNdUpdateFunctor<DeviceType::kCPU, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const {
    DoScatterNdUpdate<DeviceType::kCPU>(args.num_slices * args.slice_size, args.slice_size,
                                        args.index_ndims, args.dense_shape, indices, slices, dense);
  }
};

template<typename T, typename I>
struct ScatterNdUpdateWithStrideFunctor<DeviceType::kCPU, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices,
                  const T* slices, T* dense) const {
    DoScatterNdUpdateWithStride<DeviceType::kCPU>(args.num_slices * args.slice_size, args, indices,
                                                  slices, dense);
  }
};

template<typename T, typename I>
struct FillByNdIndexFunctor<DeviceType::kCPU, T, I> final {
  void operator()(ep::Stream* stream, const NdIndexSliceArgs& args, const I* indices, T* dense,
                  T value) const {
    DoFillByNdIndex(args.num_slices * args.slice_size, args.slice_size, args.index_ndims,
                    args.dense_shape, indices, dense, value);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ND_INDEX_SLICE_FUNCTORS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                                     BFLOAT16_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                         BOOL_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                                     BFLOAT16_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                         BOOL_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
