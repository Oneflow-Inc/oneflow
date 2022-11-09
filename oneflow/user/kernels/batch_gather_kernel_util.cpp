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
#include "oneflow/user/kernels/batch_gather_kernel_util.h"

namespace oneflow {

namespace {

Shape GetFlatShape(const ShapeView& shape, const int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
void BatchGatherForward(ep::Stream* stream, const Blob* in, const Blob* indices, Blob* out) {
  const int64_t axis = indices->shape_view().NumAxes() - 1;
  const Shape flat_out_shape = GetFlatShape(out->shape_view(), axis);
  BatchGatherKernelUtilImpl<device_type, T, K>::Forward(stream, in->dptr<T>(), indices->dptr<K>(),
                                                        flat_out_shape, in->shape_view().At(axis),
                                                        out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void BatchGatherBackward(ep::Stream* stream, const Blob* out_diff, const Blob* indices,
                         Blob* in_diff) {
  Memset<device_type>(stream, in_diff->mut_dptr<T>(), 0, in_diff->ByteSizeOfBlobBody());
  const int64_t axis = indices->shape_view().NumAxes() - 1;
  const Shape flat_out_diff_shape = GetFlatShape(out_diff->shape_view(), axis);
  BatchGatherKernelUtilImpl<device_type, T, K>::Backward(
      stream, out_diff->dptr<T>(), indices->dptr<K>(), flat_out_diff_shape,
      in_diff->shape_view().At(axis), in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct BatchGatherSwitchUtil final {
#define MAKE_BATCH_GATHER_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_BATCH_GATHER_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC(BatchGatherForward);
  DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC(BatchGatherBackward);
#undef DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC
#undef MAKE_BATCH_GATHER_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void BatchGatherKernelUtil<device_type, T>::Forward(ep::Stream* stream, const Blob* in,
                                                    const Blob* indices, Blob* out) {
  BatchGatherSwitchUtil<device_type, T>::SwitchBatchGatherForward(SwitchCase(indices->data_type()),
                                                                  stream, in, indices, out);
}

template<DeviceType device_type, typename T>
void BatchGatherKernelUtil<device_type, T>::Backward(ep::Stream* stream, const Blob* out_diff,
                                                     const Blob* indices, Blob* in_diff) {
  BatchGatherSwitchUtil<device_type, T>::SwitchBatchGatherBackward(
      SwitchCase(indices->data_type()), stream, out_diff, indices, in_diff);
}

template<typename T, typename K>
struct BatchGatherKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void Forward(ep::Stream* stream, const T* in, const K* indices,
                      const Shape& flat_out_shape, int64_t gather_dim_size, T* out);
  static void Backward(ep::Stream* stream, const T* out_diff, const K* indices,
                       const Shape& flat_out_diff_shape, int64_t gather_dim_size, T* in_diff);
};

template<typename T, typename K>
void BatchGatherKernelUtilImpl<DeviceType::kCPU, T, K>::Forward(ep::Stream* stream, const T* in,
                                                                const K* indices,
                                                                const Shape& flat_out_shape,
                                                                const int64_t gather_dim_size,
                                                                T* out) {
  const int64_t batch_num = flat_out_shape.At(0);
  const int64_t indices_num = flat_out_shape.At(1);
  const int64_t instance_size = flat_out_shape.At(2);
  FOR_RANGE(int64_t, batch_idx, 0, batch_num) {
    FOR_RANGE(int64_t, i, 0, indices_num) {
      const K idx = indices[batch_idx * indices_num + i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = in + batch_idx * gather_dim_size * instance_size + idx * instance_size;
      T* to = out + batch_idx * indices_num * instance_size + i * instance_size;
      std::copy(from, from + instance_size, to);
    }
  }
}

template<typename T, typename K>
void BatchGatherKernelUtilImpl<DeviceType::kCPU, T, K>::Backward(
    ep::Stream* stream, const T* out_diff, const K* indices, const Shape& flat_out_diff_shape,
    const int64_t gather_dim_size, T* in_diff) {
  const int64_t batch_num = flat_out_diff_shape.At(0);
  const int64_t indices_num = flat_out_diff_shape.At(1);
  const int64_t instance_size = flat_out_diff_shape.At(2);
  FOR_RANGE(int64_t, batch_idx, 0, batch_num) {
    FOR_RANGE(int64_t, i, 0, indices_num) {
      const int64_t idx = indices[batch_idx * indices_num + i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = out_diff + batch_idx * indices_num * instance_size + i * instance_size;
      T* to = in_diff + batch_idx * gather_dim_size * instance_size + idx * instance_size;
      std::transform(from, from + instance_size, to, to, std::plus<T>());
    }
  }
}

#define INSTANTIATE_BATCH_GATHER_KERNEL_UTIL_IMPL_CPU(in_type_pair, index_type_pair)          \
  template struct BatchGatherKernelUtilImpl<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                            OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BATCH_GATHER_KERNEL_UTIL_IMPL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_BATCH_GATHER_KERNEL_UTIL_IMPL_CPU

#define INSTANTIATE_BATCH_GATHER_KERNEL_UTIL(device_type, in_type_pair) \
  template struct BatchGatherKernelUtil<device_type, OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BATCH_GATHER_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_BATCH_GATHER_KERNEL_UTIL

}  // namespace oneflow
