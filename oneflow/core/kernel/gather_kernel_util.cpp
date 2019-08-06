#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

namespace {

Shape GetFlatShape(const Shape& shape, int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
void GatherForward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out) {
  const Shape flat_in_shape = GetFlatShape(in->shape(), axis);
  GatherKernelUtilImpl<device_type, T, K>::Forward(ctx, indices->dptr<K>(),
                                                   indices->shape().elem_cnt(), in->dptr<T>(),
                                                   flat_in_shape, out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void GatherBackward(DeviceCtx* ctx, const Blob* indices, const Blob* out_diff, int64_t axis,
                    Blob* in_diff) {
  const Shape flat_in_shape = GetFlatShape(in_diff->shape(), axis);
  GatherKernelUtilImpl<device_type, T, K>::Backward(
      ctx, indices->dptr<K>(), indices->shape().elem_cnt(), out_diff->dptr<T>(), flat_in_shape,
      in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct GatherSwitchUtil final {
#define MAKE_GATHER_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_GATHER_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_GATHER_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_GATHER_STATIC_SWITCH_FUNC(GatherForward);
  DEFINE_GATHER_STATIC_SWITCH_FUNC(GatherBackward);
#undef DEFINE_GATHER_STATIC_SWITCH_FUNC
#undef MAKE_GATHER_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void GatherKernelUtil<device_type, T>::Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in,
                                               const int64_t axis, Blob* out) {
  GatherSwitchUtil<device_type, T>::SwitchGatherForward(SwitchCase(indices->data_type()), ctx,
                                                        indices, in, axis, out);
}

template<DeviceType device_type, typename T>
void GatherKernelUtil<device_type, T>::Backward(DeviceCtx* ctx, const Blob* indices,
                                                const Blob* out_diff, const int64_t axis,
                                                Blob* in_diff) {
  GatherSwitchUtil<device_type, T>::SwitchGatherBackward(SwitchCase(indices->data_type()), ctx,
                                                         indices, out_diff, axis, in_diff);
}

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out);
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff);
};

template<typename T, typename K>
void GatherKernelUtilImpl<DeviceType::kCPU, T, K>::Forward(DeviceCtx* ctx, const K* indices,
                                                           int64_t num_indices, const T* in,
                                                           const Shape& flat_in_shape, T* out) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t gather_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices[i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = in + outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size;
      T* to = out + outer_idx * num_indices * inner_dim_size + i * inner_dim_size;
      std::copy(from, from + inner_dim_size, to);
    }
  }
}

template<typename T, typename K>
void GatherKernelUtilImpl<DeviceType::kCPU, T, K>::Backward(DeviceCtx* ctx, const K* indices,
                                                            int64_t num_indices, const T* out_diff,
                                                            const Shape& flat_in_shape,
                                                            T* in_diff) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t gather_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices[i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = out_diff + outer_idx * num_indices * inner_dim_size + i * inner_dim_size;
      T* to = in_diff + outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size;
      std::transform(from, from + inner_dim_size, to, to, std::plus<T>());
    }
  }
}

#define INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_CPU_IMPL

#define INITIATE_GATHER_KERNEL_UTIL(device_type, in_type_pair) \
  template struct GatherKernelUtil<device_type, OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template struct GatherKernelUtil<DeviceType::kGPU, float16>;
#endif

}  // namespace oneflow
