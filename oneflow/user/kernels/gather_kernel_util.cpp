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
#include "oneflow/user/kernels/gather_kernel_util.h"

namespace oneflow {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
void GatherForward(ep::Stream* stream, const Blob* indices, const Blob* in, int64_t axis, Blob* out,
                   const int64_t offset) {
  const Shape& flat_in_shape = GetFlatShape(in->shape_view(), axis);
  GatherKernelUtilImpl<device_type, T, K>::Forward(stream, indices->dptr<K>(),
                                                   indices->shape_view().elem_cnt(), in->dptr<T>(),
                                                   flat_in_shape, out->mut_dptr<T>(), offset);
}

template<DeviceType device_type, typename T>
struct GatherSwitchUtil final {
#define MAKE_GATHER_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_GATHER_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_GATHER_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));
  DEFINE_GATHER_STATIC_SWITCH_FUNC(GatherForward);
#undef DEFINE_GATHER_STATIC_SWITCH_FUNC
#undef MAKE_GATHER_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void GatherKernelUtil<device_type, T>::Forward(ep::Stream* stream, const Blob* indices,
                                               const Blob* in, const int64_t axis, Blob* out) {
  GatherKernelUtil<device_type, T>::Forward(stream, indices, in, axis, out, 0);
}

template<DeviceType device_type, typename T>
void GatherKernelUtil<device_type, T>::Forward(ep::Stream* stream, const Blob* indices,
                                               const Blob* in, const int64_t axis, Blob* out,
                                               const int64_t offset) {
  GatherSwitchUtil<device_type, T>::SwitchGatherForward(SwitchCase(indices->data_type()), stream,
                                                        indices, in, axis, out, offset);
}

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void Forward(ep::Stream* stream, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset);
};

template<typename T, typename K>
void GatherKernelUtilImpl<DeviceType::kCPU, T, K>::Forward(ep::Stream* stream, const K* indices,
                                                           int64_t num_indices, const T* in,
                                                           const Shape& flat_in_shape, T* out,
                                                           const int64_t offset) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t gather_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      CHECK_GE(indices[i], 0);
      const int64_t idx = indices[i] - offset;
      T* to = out + outer_idx * num_indices * inner_dim_size + i * inner_dim_size;
      if (idx >= 0 && idx < gather_dim_size) {
        const T* from = in + outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size;
        std::copy(from, from + inner_dim_size, to);
      } else {
        std::memset(reinterpret_cast<void*>(to), 0, inner_dim_size * sizeof(T));
      }
    }
  }
}

#define INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL,
                                 GATHER_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ,
                                 GATHER_INDEX_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL

#define INITIATE_GATHER_KERNEL_UTIL(device_type, in_type_pair) \
  template struct GatherKernelUtil<device_type, OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 GATHER_DATA_TYPE_SEQ);
// For cpu float16/bfloat16
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU),
                                 FLOAT16_DATA_TYPE_SEQ BFLOAT16_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL

}  // namespace oneflow
