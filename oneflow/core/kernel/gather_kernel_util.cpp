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
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include <assert.h>

namespace oneflow {

namespace {

Shape GetFlatShape(const ShapeView& shape, int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
void GatherForward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out,
                   const int64_t offset) {
  const Shape& flat_in_shape = GetFlatShape(in->shape(), axis);
  GatherKernelUtilImpl<device_type, T, K>::Forward(ctx, indices->dptr<K>(),
                                                   indices->shape().elem_cnt(), in->dptr<T>(),
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
void GatherKernelUtil<device_type, T>::Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in,
                                               const int64_t axis, Blob* out) {
  GatherKernelUtil<device_type, T>::Forward(ctx, indices, in, axis, out, 0);
}

template<DeviceType device_type, typename T>
void GatherKernelUtil<device_type, T>::Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in,
                                               const int64_t axis, Blob* out,
                                               const int64_t offset) {
  GatherSwitchUtil<device_type, T>::SwitchGatherForward(SwitchCase(indices->data_type()), ctx,
                                                        indices, in, axis, out, offset);
}

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset);
};

template<typename T, typename K>
void GatherKernelUtilImpl<DeviceType::kCPU, T, K>::Forward(DeviceCtx* ctx, const K* indices,
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
        std::memset(to, 0, inner_dim_size * sizeof(K));
      }
    }
  }
}

#define INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL

#define INITIATE_GATHER_KERNEL_UTIL(device_type, in_type_pair) \
  template struct GatherKernelUtil<device_type, OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 GATHER_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL

#if defined(WITH_HIP)

namespace {

template<typename K, typename IDX>
__device__ IDX GetInOffset(const IDX out_offset, const K* indices, const IDX num_indices,
                           const IDX gather_dim_size, const IDX inner_dim_size, const IDX offset) {
  const IDX outer_dim_elem_cnt = num_indices * inner_dim_size;
  const IDX outer_idx = out_offset / outer_dim_elem_cnt;
  const IDX indices_idx = out_offset % outer_dim_elem_cnt / inner_dim_size;
  const IDX inner_idx = out_offset % inner_dim_size;
  assert(indices[indices_idx] >= 0);
  const IDX idx = indices[indices_idx] - offset;
  if (idx >= 0 && idx < gather_dim_size) {
    return outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size + inner_idx;
  } else {
    return -1;
  }
}

template<typename T, typename K, typename IDX>
__global__ void GatherForwardGpu(const IDX elem_cnt, const K* indices, const IDX num_indices,
                                 const T* in, const IDX gather_dim_size, const IDX inner_dim_size,
                                 T* out, const IDX offset) {
  HIP_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    const IDX in_offset =
        GetInOffset<K, IDX>(i, indices, num_indices, gather_dim_size, inner_dim_size, offset);
    if (in_offset < 0) {
      out[i] = 0;
    } else {
      out[i] = in[in_offset];
    }
  }
}

bool IsSafeUseIndex32(const Shape& flat_in_shape, const int64_t num_indices) {
  const int64_t in_elem_cnt = flat_in_shape.elem_cnt();
  const int64_t out_elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

}  // namespace

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset) {
    const int64_t out_elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    if (IsSafeUseIndex32(flat_in_shape, num_indices)) {
      GatherForwardGpu<T, K, int32_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kHipThreadsNumPerBlock, 0, ctx->rocm_stream()>>>(
              out_elem_cnt, indices, num_indices, in, flat_in_shape.At(1), flat_in_shape.At(2), out,
              offset);
    } else {
      GatherForwardGpu<T, K, int64_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kHipThreadsNumPerBlock, 0, ctx->rocm_stream()>>>(
              out_elem_cnt, indices, num_indices, in, flat_in_shape.At(1), flat_in_shape.At(2), out,
              offset);
    }
  }
};

template<typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, float16, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const float16* in,
                      const Shape& flat_in_shape, float16* out, const int64_t offset) {
    GatherKernelUtilImpl<DeviceType::kGPU, half, K>::Forward(
        ctx, indices, num_indices, reinterpret_cast<const half*>(in), flat_in_shape,
        reinterpret_cast<half*>(out), offset);
  }
};

#define INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL, GATHER_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL

#endif

}  // namespace oneflow
